"""Fresh-match camera diagnostics. CPU only; no stored tracks or cloud required."""

from collections import Counter, OrderedDict
from dataclasses import replace
import csv
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, UnidentifiedImageError

from utils.colmap_undistortion import colmap_image_options, undistort_points
from utils.pose_audit_data import _colmap, index_images, sparse_directory
from utils.viewer_utils import read_model_config


class GeometryDataset:
    def __init__(self, source=None, model=None, images=None):
        config = read_model_config(model) if model else {}
        source = source or config.get("source_path")
        if not source:
            raise ValueError("Dataset source_path is missing")
        self.source = Path(source).expanduser().resolve()
        directory, suffix = sparse_directory(self.source)
        if suffix == "bin":
            cameras = _colmap.read_intrinsics_binary(str(directory / "cameras.bin"))
        else:
            cameras = {}
            for line in (directory / "cameras.txt").read_text().splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    p = line.split()
                    cameras[int(p[0])] = _colmap.Camera(int(p[0]), p[1], int(p[2]), int(p[3]), np.array(p[4:], float))
        self.calibrations = {key: colmap_image_options(camera)["colmap_calibration"] for key, camera in cameras.items()}
        self.frames = index_images(directory / f"images.{suffix}", suffix)
        if len(self.frames) < 2:
            raise ValueError("Need at least two registered images")
        # Text exports often round quaternion entries. A sign flip is harmless.
        for i, frame in enumerate(self.frames):
            norm = np.linalg.norm(frame.qvec)
            if not np.isfinite(norm) or norm < 1e-12 or not np.isfinite(frame.tvec).all():
                raise ValueError(f"Invalid pose: {frame.name}")
            rotation = _colmap.qvec2rotmat(frame.qvec / norm)
            self.frames[i] = replace(frame, qvec=frame.qvec / norm, rotation=rotation, center=-rotation.T @ frame.tvec)
            if frame.camera_id not in self.calibrations:
                raise ValueError(f"Missing calibration for {frame.name}")
        self.images = self.source / (images or config.get("images", "images"))
        basenames = [Path(frame.name).name for frame in self.frames]
        duplicates = {name for name, count in Counter(basenames).items() if count > 1}
        self.paths = []
        for frame in self.frames:
            path = self.images / frame.name
            if not path.is_file() and Path(frame.name).name not in duplicates:
                path = self.images / Path(frame.name).name
            self.paths.append(path)
        self.available = [i for i, path in enumerate(self.paths) if path.is_file()]
        if len(self.available) < 2:
            raise ValueError(f"Fewer than two registered RGB files found under {self.images}")


def pixel_calibration(calibration, size):
    """COLMAP corner coordinates -> resized image pixel-index coordinates."""
    matrix = np.array(calibration["matrix"], float)
    matrix[0] *= size[0] / calibration["width"]
    matrix[1] *= size[1] / calibration["height"]
    matrix[:2, 2] -= .5
    return matrix, np.array(calibration["distortion"], float)


def load_pixels(path, max_width):
    # Keep strict corruption handling even when another importer enables it.
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(path) as image:
            image.load()
            image = image.convert("RGB")
            if image.width > max_width:
                image = image.resize((max_width, max(1, round(image.height * max_width / image.width))), Image.Resampling.LANCZOS)
            return np.array(image)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


class FeatureStore:
    """Bounded RAM descriptors, disk cache, and one decoded image at a time."""
    def __init__(self, data, output, max_width=1600, max_features=4000):
        self.data, self.max_width, self.max_features = data, max_width, max_features
        self.directory = Path(output) / "feature_cache"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory = OrderedDict()
        self.detector = cv2.SIFT_create(nfeatures=max_features)

    def get(self, index):
        path = self.data.paths[index]
        stat = path.stat()
        signature = [2, str(path.resolve()), stat.st_size, stat.st_mtime_ns, cv2.__version__, self.max_width, self.max_features]
        digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        if digest in self.memory:
            self.memory.move_to_end(digest)
            return self.memory[digest]
        cache = self.directory / (digest + ".npz")
        result = None
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as saved:
                    xy, desc, size = saved["xy"], saved["desc"], tuple(saved["size"].tolist())
                if xy.shape == (len(desc), 2) and desc.shape == (len(xy), 128) and len(size) == 2:
                    result = xy, desc, size
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                pass
        if result is None:
            pixels = load_pixels(path, self.max_width)
            size = (pixels.shape[1], pixels.shape[0])
            keys, desc = self.detector.detectAndCompute(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), None)
            xy = np.array([key.pt for key in keys], dtype=np.float64).reshape(-1, 2)
            desc = np.empty((0, 128), np.float32) if desc is None else desc
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npz", delete=False) as stream:
                temporary = stream.name
                np.savez_compressed(stream, xy=xy, desc=desc, size=size)
            os.replace(temporary, cache)
            result = xy, desc, size
        self.memory[digest] = result
        if len(self.memory) > 4:
            self.memory.popitem(last=False)
        return result


def plan_pairs(data, samples, gaps, revisit_budget, revisit_gap, seed):
    anchors = [data.available[i] for i in np.linspace(0, len(data.available) - 1, min(samples, len(data.available)), dtype=int)]
    available = set(data.available)
    pairs = {(i, i + gap): "local" for i in anchors for gap in gaps if i + gap in available}
    candidates = [(i, j) for i, j in itertools.combinations(anchors, 2) if j - i >= revisit_gap]
    # Half the budget uses camera proximity; the other half is independent of
    # pose proximity so drift cannot entirely hide revisits from the audit.
    if len(candidates) > revisit_budget:
        near, chosen = {}, []
        for i, j in candidates:
            a, b = data.frames[i], data.frames[j]
            angle = np.arccos(np.clip((np.trace(b.rotation @ a.rotation.T) - 1) / 2, -1, 1))
            distance = float(np.linalg.norm(a.center - b.center))
            near.setdefault(i, []).append((distance * (1 + angle), (i, j)))
        ranked = {i: sorted(values) for i, values in near.items()}
        for depth in range(len(anchors)):
            for i in sorted(ranked):
                if depth < len(ranked[i]) and len(chosen) < revisit_budget // 2:
                    chosen.append(ranked[i][depth][1])
            if len(chosen) >= revisit_budget // 2:
                break
        remaining = [pair for pair in candidates if pair not in set(chosen)]
        rng = np.random.default_rng(seed)
        chosen += [remaining[int(k)] for k in rng.choice(len(remaining), min(revisit_budget - len(chosen), len(remaining)), replace=False)]
        candidates = chosen
    for pair in candidates[:revisit_budget]:
        pairs.setdefault(pair, "revisit")
    return [(i, j, kind) for (i, j), kind in sorted(pairs.items())]


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None)


def sampson(F, a, b):
    aa, bb = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    l2, l1 = aa @ F.T, bb @ F
    denominator = np.sum(l2[:, :2] ** 2, axis=1) + np.sum(l1[:, :2] ** 2, axis=1)
    return np.divide(abs(np.sum(bb * l2, axis=1)), np.sqrt(denominator),
                     out=np.full(len(a), np.nan), where=denominator > 1e-30)


def pose_fundamental(a, b, Ka, Kb):
    rotation = b.rotation @ a.rotation.T
    translation = b.tvec - rotation @ a.tvec
    norm = np.linalg.norm(translation)
    if norm < 1e-10:
        return None
    x, y, z = translation / norm
    essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ rotation
    return np.linalg.inv(Kb).T @ essential @ np.linalg.inv(Ka)


def coverage(xy, size):
    normalized = xy / size
    cells = np.floor(normalized * 4).clip(0, 3).astype(int)
    return len(np.unique(cells, axis=0)), float(np.sqrt(max(0, np.linalg.eigvalsh(np.cov(normalized.T))[0])))


def verify_matches(a, b, da, db, sizes, ratio=.65, threshold=1.5, minimum=24, seed=42):
    """Independent image geometry, never gated by agreement with saved poses."""
    if min(len(da), len(db)) < minimum:
        return None, "too_few_features"
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(da, db, k=2) if m.distance < ratio * n.distance}
    backward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(db, da, k=2) if m.distance < ratio * n.distance}
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(pairs) < minimum:
        return None, "too_few_mutual_matches"
    aa, bb = np.array([a[i] for i, j in pairs]), np.array([b[j] for i, j in pairs])
    # SIFT can emit several orientations at the same image point. Count that
    # correspondence once rather than inflating support with duplicates.
    _, unique = np.unique(np.round(np.c_[aa, bb]), axis=0, return_index=True)
    aa, bb = aa[np.sort(unique)], bb[np.sort(unique)]
    if len(aa) < minimum:
        return None, "too_few_unique_matches"
    cv2.setRNGSeed(int(seed % (2**31 - 1)))
    F, mask = cv2.findFundamentalMat(aa, bb, cv2.USAC_MAGSAC, threshold, .999, 10000)
    if F is None or mask is None or not np.isfinite(F).all():
        return None, "fundamental_fit_failed"
    valid = mask.ravel().astype(bool)
    if valid.sum() < minimum or valid.mean() < .5:
        return None, "weak_geometric_support"
    aa, bb = aa[valid], bb[valid]
    refined, _ = cv2.findFundamentalMat(aa, bb, cv2.FM_8POINT)
    if refined is not None and refined.shape == (3, 3) and np.isfinite(refined).all():
        F = refined
    support = [coverage(xy, size) for xy, size in zip((aa, bb), sizes)]
    if min(cells for cells, spread in support) < 4 or min(spread for cells, spread in support) < .02:
        return None, "localized_or_collinear_matches"
    homography, hmask = cv2.findHomography(aa, bb, cv2.RANSAC, threshold)
    hratio = float(hmask.mean()) if hmask is not None else 0.
    held_out = []
    rng = np.random.default_rng(seed)
    for _ in range(12):
        indices = rng.permutation(len(aa))
        split = max(8, int(len(aa) * .7))
        fitted, _ = cv2.findFundamentalMat(aa[indices[:split]], bb[indices[:split]], cv2.FM_8POINT)
        if fitted is not None and fitted.shape == (3, 3):
            held_out.extend(sampson(fitted, aa[indices[split:]], bb[indices[split:]]))
    return dict(a=aa, b=bb, F=F, mutual_unique=len(valid), inlier_fraction=float(valid.mean()),
                coverage_cells=[s[0] for s in support], homography_fraction=hratio,
                fitted=stats(sampson(F, aa, bb)), held_out=stats(held_out)), None


def match_pair(data, store, i, j, args):
    a, da, size_a = store.get(i)
    b, db, size_b = store.get(j)
    Ka, D_a = pixel_calibration(data.calibrations[data.frames[i].camera_id], size_a)
    Kb, D_b = pixel_calibration(data.calibrations[data.frames[j].camera_id], size_b)
    a = undistort_points(a[:, None], Ka, D_a, Ka).reshape(-1, 2) if len(a) else a
    b = undistort_points(b[:, None], Kb, D_b, Kb).reshape(-1, 2) if len(b) else b
    match, reason = verify_matches(a, b, da, db, (size_a, size_b), args.ratio, args.ransac_px, args.min_matches, args.seed + i + j)
    if match is None:
        return None, reason
    return {**match, "Ka": Ka, "Kb": Kb, "size_a": size_a, "size_b": size_b}, None


def parallax_degrees(a, b, Ka, Kb, rotation):
    rays_a = np.c_[a, np.ones(len(a))] @ np.linalg.inv(Ka).T @ rotation.T
    rays_b = np.c_[b, np.ones(len(b))] @ np.linalg.inv(Kb).T
    cosine = np.sum(rays_a * rays_b, axis=1) / (np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1))
    return float(np.median(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))


def offset_scores(frames, i, j, match, limit):
    """Same image correspondences and intrinsics at every candidate offset."""
    if i - limit < 0 or j + limit >= len(frames):
        return None
    if any(frames[k + offset].camera_id != frames[k].camera_id
           for k in (i, j) for offset in range(-limit, limit + 1)):
        return None
    scores = {}
    for offset in range(-limit, limit + 1):
        F = pose_fundamental(frames[i + offset], frames[j + offset], match["Ka"], match["Kb"])
        if F is None:
            return None
        error = stats(sampson(F, match["a"], match["b"]))["median"]
        if error is None:
            return None
        scores[str(offset)] = error
    return scores


def sequence_block(i, j, count, limit):
    """Four disjoint temporal blocks; all shifted camera support stays inside."""
    first, last = 4 * (i - limit) // count, 4 * (j + limit) // count
    return int(first) if first == last and 0 <= first < 4 else None


def summarize_offsets(rows, limit):
    eligible = [row for row in rows if row.get("offset_scores") is not None and row.get("sequence_block") is not None]
    discovery = [row for row in eligible if row["sequence_block"] % 2 == 0]
    validation = [row for row in eligible if row["sequence_block"] % 2 == 1]
    result = dict(status="unavailable", discovery_pairs=len(discovery), validation_pairs=len(validation),
                  convention="+d means image i is compared using pose i+d in natural registered-image order; not a timestamp in seconds",
                  explanation="Need at least 5 informative local pairs in discovery and 5 in disjoint validation blocks.")
    if len(discovery) < 5 or len(validation) < 5:
        return result
    sweep = []
    for offset in range(-limit, limit + 1):
        sweep.append(dict(offset=offset,
                          discovery_median_px=stats([row["offset_scores"][str(offset)] for row in discovery])["median"],
                          validation_median_px=stats([row["offset_scores"][str(offset)] for row in validation])["median"]))
    best = min(sweep, key=lambda row: (row["discovery_median_px"], abs(row["offset"])))
    zero = next(row for row in sweep if row["offset"] == 0)
    differences = [row["offset_scores"]["0"] - row["offset_scores"][str(best["offset"])] for row in validation]
    per_block = {str(block): stats([row["offset_scores"]["0"] - row["offset_scores"][str(best["offset"])]
                                  for row in validation if row["sequence_block"] == block])["median"]
                 for block in (1, 3)}
    improvement = zero["validation_median_px"] - best["validation_median_px"]
    supported = (best["offset"] != 0 and abs(best["offset"]) < limit and improvement > .1
                 and best["validation_median_px"] < .8 * zero["validation_median_px"]
                 and all(value is not None and value > 0 for value in per_block.values()))
    result.update(status="candidate_offset" if supported else "inconclusive" if best["offset"] else "no_offset_supported",
                  selected_offset=best["offset"], sweep=sweep, validation_pair_improvement_px=stats(differences),
                  validation_block_improvements_px=per_block,
                  explanation=("Offset chosen on discovery blocks improves both held-out blocks. Verify timestamps before applying it."
                               if supported else "No nonzero offset passed the held-out consistency checks. This does not prove synchronization is correct."))
    if best["offset"] and abs(best["offset"]) == limit:
        result["explanation"] = "The discovery minimum is at the sweep boundary; extend --max_offset before interpreting it."
    return result


def error_profile(errors, coordinates, size):
    normalized = coordinates / size
    radius = np.linalg.norm((normalized - .5) * 2, axis=1) / np.sqrt(2)
    return dict(row_quarters=[stats(errors[(normalized[:, 1] >= k / 4) & (normalized[:, 1] < (k + 1) / 4)]) for k in range(4)],
                radius_thirds=[stats(errors[(radius >= k / 3) & (radius < (k + 1) / 3)]) for k in range(3)])


def summary_text(result):
    lines = [f"Fresh-match camera check: {result['measured_pairs']} measured / {result['attempted_pairs']} candidate pairs.",
             f"RGB available: {result['available_images']:,} / {result['registered_images']:,} registered views.",
             "Metric: square-root Sampson epipolar distance in resized, undistorted analysis pixels."]
    for kind in ("local", "revisit"):
        record = result["groups"][kind]
        value = record["median"]
        lines.append(f"{kind.capitalize()}: {record['count']} pairs; median of pair medians = " + (f"{value:.3f} px" if value is not None else "unavailable"))
    rejected = Counter(row.get("reason", row["status"]) for row in result["pairs"] if row["status"] != "measured")
    if rejected:
        lines.append("Unmeasured candidates: " + "; ".join(f"{reason}: {count}" for reason, count in rejected.most_common()))
    timing = result["synchronization"]
    lines.append(f"Synchronization: {timing['status']}. {timing['explanation']}")
    if "selected_offset" in timing:
        lines.append(f"Discovery-selected pose offset: {timing['selected_offset']:+d} sequence entries; no dataset changes were made.")
    lines += ["Epipolar consistency is not absolute pose accuracy or stored-landmark reprojection error.",
              "Candidate revisits require visual confirmation of static scene overlap; repetitive texture and screens can produce false matches.",
              "Homography-dominated matches do not uniquely constrain 3D pose; they are excluded from synchronization scoring."]
    if all(not np.any(cal["distortion"]) for cal in result["calibrations"].values()):
        lines.append("All calibrations have zero stored distortion coefficients; this does not establish that the original lens distortion was zero.")
    return "\n".join(lines)


def preview(data, store, i, j, match, destination):
    """Observed correspondences and nearest saved-pose epipolar-line points."""
    pixels = []
    for index, matrix in ((i, match["Ka"]), (j, match["Kb"])):
        image = load_pixels(data.paths[index], store.max_width)
        distortion = np.array(data.calibrations[data.frames[index].camera_id]["distortion"])
        if np.any(distortion):
            image = cv2.undistort(image, matrix, distortion, None, matrix)
        pixels.append(Image.fromarray(image))
    F = pose_fundamental(data.frames[i], data.frames[j], match["Ka"], match["Kb"])
    chosen = np.linspace(0, len(match["a"]) - 1, min(20, len(match["a"])), dtype=int)
    for number, k in enumerate(chosen, 1):
        for image, xy in zip(pixels, (match["a"][k], match["b"][k])):
            draw = ImageDraw.Draw(image)
            x, y = xy
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 235, 255), width=2)
            draw.text((x + 5, y + 3), str(number), fill=(0, 235, 255), stroke_width=1, stroke_fill="black")
        if F is not None:
            line = F @ np.r_[match["a"][k], 1]
            x, y = match["b"][k]
            denominator = line[:2] @ line[:2]
            if denominator > 1e-20:
                u, v = np.array([x, y]) - line[:2] * (line @ [x, y, 1]) / denominator
                if np.isfinite([u, v]).all():
                    draw = ImageDraw.Draw(pixels[1])
                    draw.line((x, y, u, v), fill=(255, 65, 95), width=2)
                    draw.line((u - 4, v, u + 4, v), fill=(255, 180, 50), width=2)
                    draw.line((u, v - 4, u, v + 4), fill=(255, 180, 50), width=2)
    canvas = Image.new("RGB", (sum(image.width for image in pixels), max(image.height for image in pixels)), (15, 20, 27))
    canvas.paste(pixels[0], (0, 0))
    canvas.paste(pixels[1], (pixels[0].width, 0))
    canvas.save(destination, quality=92)


def write_report(data, store, result, args):
    output = args.output
    text = summary_text(result)
    (output / "summary.txt").write_text(text + "\n")
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    measured = [row for row in result["pairs"] if row["status"] == "measured"]
    fields = ["image_a", "image_b", "kind", "status", "matches", "saved_median_px", "saved_p95_px", "fitted_median_px", "held_out_fitted_median_px", "homography_fraction", "parallax_degrees", "reason"]
    with (output / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["pairs"])
    selected = []
    for kind in ("local", "revisit"):
        ranked = sorted([row for row in measured if row["kind"] == kind], key=lambda row: row["saved_median_px"], reverse=True)
        # Show both high and low residual examples, avoiding worst-only sampling.
        ordered = ranked[::2] + list(reversed(ranked[1::2]))
        selected += ordered[:max(1, args.previews // 2)]
    selected = selected[:args.previews]
    names = ["index.html", "summary.txt", "results.json", "pairs.csv", "cameras.npz"]
    previews = {}
    for number, row in enumerate(selected):
        name = f"pair_{number:03d}.jpg"
        with np.load(output / row["correspondences"], allow_pickle=False) as saved:
            match = {key: saved[key] for key in saved.files}
        preview(data, store, row["a"], row["b"], match, output / name)
        names.append(name)
        previews[(row["a"], row["b"])] = name
    table, cards = [], []
    def value(number):
        return "—" if number is None else f"{number:.3f}"
    for row in measured:
        pair = html.escape(row["image_a"] + " ↔ " + row["image_b"])
        warning = " · homography dominated" if row["homography_fraction"] >= .9 else ""
        table.append(f"<tr><td>{pair}</td><td>{row['kind']}{warning}</td><td>{row['matches']}</td><td>{value(row['saved_median_px'])}</td><td>{value(row['saved_p95_px'])}</td><td>{value(row['held_out_fitted_median_px'])}</td></tr>")
        if (row["a"], row["b"]) in previews:
            name = previews[(row["a"], row["b"])]
            cards.append(f'<article><h2>{pair}</h2><p>{row["kind"]}{warning} · saved median {value(row["saved_median_px"])} px. Click to inspect full pixels.</p><a href="{name}"><img src="{name}"></a></article>')
        names.append(row["correspondences"])
    timing = result["synchronization"]
    timing_rows = "".join(f"<tr><td>{row['offset']:+d}</td><td>{row['discovery_median_px']:.3f}</td><td>{row['validation_median_px']:.3f}</td></tr>" for row in timing.get("sweep", []))
    page = ('<!doctype html><meta charset="utf-8"><title>Camera geometry check</title>'
            '<style>body{font:15px system-ui;background:#111820;color:#e5edf5;margin:24px;max-width:1500px}pre{white-space:pre-wrap;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px 12px;border-bottom:1px solid #354354;text-align:left}img{width:100%}article{margin:30px 0}a{color:#71ccff}</style>'
            '<h1>Camera geometry check</h1><pre>' + html.escape(text) + '</pre>'
            '<p>Image-fit errors are repeated held-out fits within the image-verified match set. They are conditional consistency checks, not ground truth. '
            'Revisit candidates use both pose proximity and sequence-wide sampling. Results do not automatically label poses good or bad.</p>'
            '<h2>Measured pairs</h2><table><tr><th>Images</th><th>Group</th><th>Matches</th><th>Saved median px</th><th>Saved p95 px</th><th>Held-out image-fit median px</th></tr>' + ''.join(table) + '</table>'
            '<h2>Pose offset sweep</h2><p>' + html.escape(timing["convention"]) + '</p>'
            '<p>Offsets use identical pairs and correspondences. Discovery blocks 0/2 choose the offset; blocks 1/3 evaluate it. No shifted pose support crosses block boundaries. Homography-dominated or low-parallax pairs are omitted.</p>'
            '<table><tr><th>Offset</th><th>Discovery median px</th><th>Validation median px</th></tr>' + timing_rows + '</table>'
            '<h2>Correspondence previews</h2><p>Cyan numbered circles identify matching features in both photos. In the right photo, '
            'orange crosses mark the nearest point on the epipolar line predicted by the saved cameras; red lines show the distance. '
            'These crosses are not projections of the original point cloud. Images use the undistorted analysis grid.</p>' + ''.join(cards))
    (output / "index.html").write_text(page)
    np.savez_compressed(output / "cameras.npz", names=np.array([frame.name for frame in data.frames]),
                        camera_ids=np.array([frame.camera_id for frame in data.frames]),
                        rotation=np.array([frame.rotation for frame in data.frames]), translation=np.array([frame.tvec for frame in data.frames]))
    archive = output.parent / (output.name + "_results.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(output / name, output.name + "/" + name)
    return archive


def run_check(data, args):
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matches").mkdir(exist_ok=True)
    cv2.setNumThreads(args.threads)
    store = FeatureStore(data, args.output, args.max_width, args.max_features)
    planned = plan_pairs(data, args.samples, args.local_gaps, args.revisit_pairs, args.revisit_gap, args.seed)
    print(f"Registered {len(data.frames):,} poses; {len(data.available):,} RGB files available. Checking {len(planned)} pairs on CPU.", flush=True)
    rows = []
    for number, (i, j, kind) in enumerate(planned):
        row = dict(a=i, b=j, image_a=data.frames[i].name, image_b=data.frames[j].name, kind=kind, gap=j-i)
        try:
            match, reason = match_pair(data, store, i, j, args)
        except (UnidentifiedImageError, FileNotFoundError) as error:
            match, reason = None, f"missing_or_corrupt_image: {error}"
        except OSError as error:
            if error.errno is not None:
                raise
            match, reason = None, f"corrupt_image: {error}"
        if match is None:
            row.update(status="rejected", reason=reason)
        else:
            F = pose_fundamental(data.frames[i], data.frames[j], match["Ka"], match["Kb"])
            if F is None:
                row.update(status="unavailable", reason="zero_translation_epipolar_geometry_is_degenerate")
            else:
                errors = sampson(F, match["a"], match["b"])
                saved = stats(errors)
                rotation = data.frames[j].rotation @ data.frames[i].rotation.T
                parallax = parallax_degrees(match["a"], match["b"], match["Ka"], match["Kb"], rotation)
                row.update(status="measured", matches=len(errors), saved_median_px=saved["median"], saved_p95_px=saved["p95"],
                           fitted_median_px=match["fitted"]["median"], held_out_fitted_median_px=match["held_out"]["median"],
                           homography_fraction=match["homography_fraction"], coverage_cells=match["coverage_cells"],
                           analysis_size_a=match["size_a"], analysis_size_b=match["size_b"], parallax_degrees=parallax,
                           rotation_degrees=float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))),
                           profile_a=error_profile(errors, match["a"], match["size_a"]),
                           profile_b=error_profile(errors, match["b"], match["size_b"]))
                row["offset_scores"] = (offset_scores(data.frames, i, j, match, args.max_offset)
                                         if kind == "local" and match["homography_fraction"] < .9 and parallax >= .3 else None)
                row["sequence_block"] = sequence_block(i, j, len(data.frames), args.max_offset)
                filename = f"matches/{number:05d}.npz"
                np.savez_compressed(args.output / filename, a=match["a"], b=match["b"], Ka=match["Ka"], Kb=match["Kb"],
                                    fitted_F=match["F"], saved_F=F, errors=errors, size_a=match["size_a"], size_b=match["size_b"])
                row["correspondences"] = filename
        rows.append(row)
        if number % 8 == 0 or number == len(planned) - 1:
            print(f"Matched {number+1}/{len(planned)} pairs; measured {sum(row['status']=='measured' for row in rows)}.", flush=True)
    measured = [row for row in rows if row["status"] == "measured"]
    result = dict(version=1, parameters={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  units="resized undistorted analysis pixels; square-root Sampson distance", registered_images=len(data.frames),
                  available_images=len(data.available), attempted_pairs=len(rows), measured_pairs=len(measured),
                  calibrations=data.calibrations,
                  groups={kind: stats([row["saved_median_px"] for row in measured if row["kind"] == kind]) for kind in ("local", "revisit")},
                  synchronization=summarize_offsets(measured, args.max_offset), pairs=rows)
    print(summary_text(result), flush=True)
    archive = write_report(data, store, result, args)
    print(f"Report: {args.output / 'index.html'}\nSEND THIS FILE: {archive}", flush=True)
    return result
