"""Automatic, streaming reprojection analysis; never decodes the RGB dataset."""

import csv
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import cv2
import numpy as np

from utils.pose_audit_data import _statistics, project_points


def _fingerprint(data):
    paths = [data.images_file, data.directory / f"cameras.{data.suffix}"]
    paths.append(next(path for path in (data.directory / "points3D.bin", data.directory / "points3D.txt") if path.is_file()))
    records = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    records.append([(frame.id, frame.split) for frame in data.frames])
    return hashlib.sha256(json.dumps([1, cv2.__version__, records]).encode()).hexdigest()


def _record(data, index):
    frame = data.frames[index]
    calibration = data.options[frame.camera_id]["colmap_calibration"]
    xys, ids, locations, matched = data.frame_matches(index)
    xyz = data.xyz[locations[matched]] if len(data.ids) else np.empty((0, 3))
    projected, camera_xyz = project_points(xyz, frame, calibration)
    observed = xys[matched]
    valid = (camera_xyz[:, 2] > 0) & np.isfinite(camera_xyz).all(axis=1)
    valid &= np.isfinite(projected).all(axis=1) & np.isfinite(observed).all(axis=1)
    errors = np.linalg.norm(projected[valid] - observed[valid], axis=1)
    measured = len(errors)
    in_image = observed[valid]
    size = np.array([calibration["width"], calibration["height"]])
    in_image = in_image[((in_image >= 0) & (in_image < size)).all(axis=1)]
    cells = np.floor(in_image / size * 4).astype(int)
    coverage = len(np.unique(cells[:, 1] * 4 + cells[:, 0])) / 16 if len(cells) else 0.
    status = ("measured" if measured else "no_features" if not len(ids) else
              "no_triangulated_ids" if not (ids >= 0).any() else
              "missing_point_ids" if not matched.any() else "no_valid_projections")
    record = dict(index=index, image=frame.name, split=frame.split, camera_id=frame.camera_id,
                  status=status, features=len(ids), triangulated_ids=int((ids >= 0).sum()),
                  matched_points=int(matched.sum()), missing_point_ids=int(((ids >= 0) & ~matched).sum()),
                  measured_points=measured, occupied_grid_fraction=coverage,
                  calibration_size=size.tolist(), **_statistics(errors))
    record["rmse"] = float(np.sqrt(np.mean(errors ** 2))) if measured else None
    return record


def summary_text(analysis):
    summary = analysis["summary"]
    total, measured = summary["total_views"], summary["measured_views"]
    lines = [f"Reprojection audit: {measured:,} / {total:,} views have measurable 2D–3D correspondences."]
    if not measured:
        lines.append("NUMERIC REPROJECTION UNAVAILABLE: " + summary["explanation"])
        lines.append("The projected cloud is a visual check only; zero measurements do not mean zero error.")
    else:
        lines.append(f"Median of per-view median errors: {summary['median_view_error_px']:.3f} px")
        lines.append(f"Observation-weighted RMSE: {summary['observation_rmse_px']:.3f} px")
        lines.append("Errors use original COLMAP calibration pixels; they measure internal consistency, not absolute pose accuracy.")
        for index in analysis["ranked_indices"][:5]:
            row = analysis["views"][index]
            lines.append(f"  {row['image']}: median {row['median']:.3f}, p95 {row['p95']:.3f} px "
                         f"({row['measured_points']:,} points, {row['occupied_grid_fraction']:.0%} grid coverage)")
    return "\n".join(lines)


def analyze_reprojection(data, output=None, progress=None, refresh=False):
    """Scan all feature records, retaining only per-view scalar statistics."""
    output = Path(output) if output else None
    fingerprint = _fingerprint(data)
    cached = output / "reprojection.json" if output else None
    if cached and cached.is_file() and not refresh:
        try:
            analysis = json.loads(cached.read_text())
            if analysis["fingerprint"] == fingerprint:
                data.association_counts.update({row["index"]: row["matched_points"] for row in analysis["views"]})
                if progress:
                    progress("Using cached reprojection analysis.")
                return analysis
        except (ValueError, KeyError):
            pass
    records, last_update = [], time.monotonic()
    for index, frame in enumerate(data.frames):
        records.append(_record(data, index))
        if progress and time.monotonic() - last_update >= 2:
            progress(f"Checking feature records: {index + 1:,}/{len(data.frames):,} views (no RGB decoding)")
            last_update = time.monotonic()
    measured = [row for row in records if row["measured_points"]]
    ranked = sorted(measured, key=lambda row: row["median"], reverse=True)
    counts = {status: sum(row["status"] == status for row in records) for status in
              ("measured", "no_features", "no_triangulated_ids", "missing_point_ids", "no_valid_projections")}
    total_points = sum(row["measured_points"] for row in measured)
    explanation = ("The images model contains no stored 2D feature observations." if counts["no_features"] == len(records)
                   else "No usable associations between stored 2D observations and loaded 3D points were found; see per-view statuses.")
    summary = dict(total_views=len(records), measured_views=len(measured), unavailable_views=len(records) - len(measured),
                   measured_points=total_points, status_counts=counts,
                   median_view_error_px=float(np.median([row["median"] for row in measured])) if measured else None,
                   observation_rmse_px=float(np.sqrt(sum(row["rmse"] ** 2 * row["measured_points"] for row in measured) / total_points)) if total_points else None,
                   explanation=explanation if not measured else "Errors describe agreement of stored observations with the reconstructed cameras and landmarks.")
    analysis = dict(fingerprint=fingerprint, units="original COLMAP calibration pixels", summary=summary,
                    ranked_indices=[row["index"] for row in ranked], views=records)
    if output:
        output.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
        with (output / "reprojection.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (output / "reprojection.txt").write_text(summary_text(analysis) + "\n")
    return analysis


def report_indices(analysis, count):
    """Show high-residual views and a sequence-wide sample, without duplicates."""
    views = analysis["views"]
    count = min(count, len(views))
    ranked = analysis["ranked_indices"][:max(1, count // 2)]
    selected = set(ranked)
    remaining = [row["index"] for row in views if row["index"] not in selected]
    slots = count - len(ranked)
    evenly = [remaining[item] for item in np.linspace(0, len(remaining) - 1, slots, dtype=int)] if slots else []
    return ranked + evenly


def package_report(output):
    output = Path(output)
    archive = output.parent / f"{output.name}_results.zip"
    names = {"index.html", "summary.json", "reprojection.json", "reprojection.csv", "reprojection.txt"}
    names.update(re.findall(r'href="(\d+_[a-z]+\.png)"', (output / "index.html").read_text()))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(names):
            path = output / name
            if path.is_file():
                bundle.write(path, f"pose_audit/{path.name}")
    return archive
