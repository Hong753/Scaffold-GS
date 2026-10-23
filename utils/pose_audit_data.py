"""Bounded-memory COLMAP indexing and overlays for the pose audit viewer."""

from dataclasses import dataclass
import html
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.colmap_undistortion import colmap_image_options, rectification_maps, undistort_points
from utils.viewer_utils import matrix_quaternion, read_model_config


# Importing scene.colmap_loader normally executes scene/__init__.py, which
# imports CUDA extensions. Reuse the actual reader without importing Scene.
_spec = importlib.util.spec_from_file_location(
    "_pose_colmap_loader", Path(__file__).resolve().parents[1] / "scene/colmap_loader.py")
_colmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_colmap)

COLORS = {"train": (54, 211, 153), "test": (192, 132, 252),
          "unused": (120, 131, 148), "source": (120, 131, 148)}


def sequence_key(name):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", name))


def sparse_directory(source):
    for path in (source / "sparse", source / "sparse/0"):
        for suffix in ("bin", "txt"):
            if all((path / f"{name}.{suffix}").is_file() for name in ("cameras", "images")):
                return path, suffix
    raise FileNotFoundError(f"No COLMAP camera/image pair under {source}/sparse or sparse/0")


@dataclass
class AuditFrame:
    id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    offset: int
    count: int
    rotation: np.ndarray
    center: np.ndarray
    split: str = "unused"

    @property
    def wxyz(self):
        return matrix_quaternion(self.rotation.T)


def _frame(image_id, name, camera_id, qvec, tvec, offset, count):
    rotation = _colmap.qvec2rotmat(qvec)
    return AuditFrame(image_id, name, camera_id, np.asarray(qvec), np.asarray(tvec),
                      offset, count, rotation, -rotation.T @ tvec)


def index_images(path, suffix):
    """Index camera metadata and offsets; do not retain all 2D feature arrays."""
    frames = []
    with path.open("rb") as stream:
        if suffix == "bin":
            count = struct.unpack("<Q", stream.read(8))[0]
            for _ in range(count):
                record = struct.unpack("<idddddddi", stream.read(64))
                name = bytearray()
                while True:
                    value = stream.read(1)
                    if not value:
                        raise ValueError("Truncated COLMAP images.bin name")
                    if value == b"\0":
                        break
                    name.extend(value)
                points = struct.unpack("<Q", stream.read(8))[0]
                frames.append(_frame(record[0], name.decode("utf-8"), record[-1],
                                     np.array(record[1:5]), np.array(record[5:8]), stream.tell(), points))
                stream.seek(24 * points, 1)
        else:
            while True:
                line = stream.readline()
                if not line:
                    break
                if not line.strip() or line.lstrip().startswith(b"#"):
                    continue
                fields = line.decode("utf-8").split(maxsplit=9)
                offset = stream.tell()
                observations = stream.readline().split()
                if len(observations) % 3:
                    raise ValueError("Invalid COLMAP images.txt observation line")
                frames.append(_frame(int(fields[0]), fields[9].strip(), int(fields[8]),
                                     np.array(fields[1:5], float), np.array(fields[5:8], float),
                                     offset, len(observations) // 3))
    return sorted(frames, key=lambda frame: sequence_key(frame.name))


def read_observations(path, suffix, frame):
    with path.open("rb") as stream:
        stream.seek(frame.offset)
        if suffix == "bin":
            dtype = np.dtype([("xy", "<f8", (2,)), ("id", "<i8")])
            records = np.fromfile(stream, dtype=dtype, count=frame.count)
            if len(records) != frame.count:
                raise ValueError(f"Truncated observations for {frame.name}")
            return records["xy"].copy(), records["id"].copy()
        fields = stream.readline().split()
        xys = np.array([(float(fields[i]), float(fields[i + 1]))
                        for i in range(0, len(fields), 3)], dtype=np.float64).reshape(-1, 2)
        return xys, np.array([int(value) for value in fields[2::3]], dtype=np.int64)


def read_cloud(directory):
    """Retain IDs/XYZ/RGB, skipping potentially large feature-track lists."""
    path = directory / "points3D.bin"
    if path.is_file():
        with path.open("rb") as stream:
            count = struct.unpack("<Q", stream.read(8))[0]
            ids, xyz, rgb = np.empty(count, np.int64), np.empty((count, 3)), np.empty((count, 3), np.uint8)
            for i in range(count):
                record = struct.unpack("<QdddBBBd", stream.read(43))
                ids[i], xyz[i], rgb[i] = record[0], record[1:4], record[4:7]
                tracks = struct.unpack("<Q", stream.read(8))[0]
                stream.seek(tracks * 8, 1)
    else:
        path = directory / "points3D.txt"
        if not path.is_file():
            raise FileNotFoundError("Pose overlays need points3D.bin or points3D.txt with point IDs; PLY alone is insufficient")
        with path.open() as stream:
            count = sum(1 for line in stream if line.strip() and not line.lstrip().startswith("#"))
        ids, xyz, rgb = np.empty(count, np.int64), np.empty((count, 3)), np.empty((count, 3), np.uint8)
        with path.open() as stream:
            i = 0
            for line in stream:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                record = line.split(maxsplit=8)
                ids[i], xyz[i], rgb[i] = int(record[0]), list(map(float, record[1:4])), list(map(int, record[4:7]))
                i += 1
    order = np.argsort(ids)
    return ids[order], xyz[order], rgb[order]


def image_resolution(size, requested):
    width, height = size
    if requested in (1, 2, 4, 8):
        return round(width / requested), round(height / requested)
    if requested == -1:
        scale = width / 1600 if width > 1600 else 1
        return int(width / scale), int(height / scale)
    if requested < 1:
        raise ValueError("Resolution must be -1, a positive target width, or a supported downscale factor")
    scale = width / requested
    return int(width / scale), max(1, int(height / scale))


def _statistics(errors):
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return {"median": None, "p95": None, "max": None}
    return {"median": float(np.median(errors)), "p95": float(np.percentile(errors, 95)),
            "max": float(errors.max())}


def project_points(xyz, frame, calibration):
    camera_xyz = xyz @ frame.rotation.T + frame.tvec
    if not len(xyz):
        return np.empty((0, 2)), camera_xyz
    projected, _ = cv2.projectPoints(camera_xyz, np.zeros(3), np.zeros(3),
                                     np.array(calibration["matrix"]), np.array(calibration["distortion"]))
    return projected.reshape(-1, 2), camera_xyz


def overlay_image(pixels, observed, projected, limit=500, mode="Both"):
    image = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(image)
    finite = np.isfinite(observed).all(axis=1) & np.isfinite(projected).all(axis=1)
    # Keep drawing bounded, and omit pairs entirely outside the image.
    width, height = image.size
    finite &= (observed[:, 0] >= 0) & (observed[:, 0] < width) & (observed[:, 1] >= 0) & (observed[:, 1] < height)
    indices = np.flatnonzero(finite)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]
    for i in indices:
        x, y = observed[i]
        u, v = projected[i]
        if mode == "Both" and abs(u) < width * 3 and abs(v) < height * 3:
            draw.line((x, y, u, v), fill=(255, 83, 105), width=1)
        if mode in ("Both", "Observations"):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(0, 230, 255), width=1)
        if mode in ("Both", "Projections") and -5 < u < width + 5 and -5 < v < height + 5:
            draw.line((u - 3, v, u + 3, v), fill=(255, 177, 49), width=1)
            draw.line((u, v - 3, u, v + 3), fill=(255, 177, 49), width=1)
    return np.asarray(image)


class PoseAuditData:
    def __init__(self, model_path=None, source_path=None, resolution=None):
        self.root = Path(model_path).expanduser().resolve() if model_path else None
        self.config = read_model_config(self.root) if self.root else {}
        source = source_path or self.config.get("source_path")
        if not source:
            raise ValueError("Supply -m RUN_DIRECTORY or -s COLMAP_DATASET")
        self.source = Path(source).expanduser().resolve()
        if not self.source.is_dir():
            raise FileNotFoundError(f"Dataset is unavailable: {self.source}; override with -s /path/to/dataset")
        self.directory, self.suffix = sparse_directory(self.source)
        if self.suffix == "bin":
            self.cameras = _colmap.read_intrinsics_binary(str(self.directory / "cameras.bin"))
        else:
            self.cameras = {}
            for line in (self.directory / "cameras.txt").read_text().splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    p = line.split()
                    self.cameras[int(p[0])] = _colmap.Camera(int(p[0]), p[1], int(p[2]), int(p[3]), np.array(p[4:], float))
        self.images_file = self.directory / f"images.{self.suffix}"
        self.frames = index_images(self.images_file, self.suffix)
        if not self.frames:
            raise ValueError("COLMAP reconstruction has no registered images")
        self.ids, self.xyz, self.rgb = read_cloud(self.directory)
        self.images_directory = self.source / self.config.get("images", "images")
        self.resolution = resolution if resolution is not None else self.config.get("resolution", -1)
        self.undistorted = bool(self.config.get("undistort_images", False))
        self.options = {key: colmap_image_options(camera, self.undistorted,
                        self.config.get("undistort_cache_dir", "outputs/undistorted")) for key, camera in self.cameras.items()}
        # Match the original loader's sorted holdout split, then the saved subset.
        lexical = sorted(self.frames, key=lambda item: Path(item.name).name.split(".")[0])
        test_ids = set()
        if self.config.get("eval", False):
            lod = self.config.get("lod", 0)
            test_ids = {frame.id for i, frame in enumerate(lexical)
                        if (i % 8 == 0 if lod <= 0 else (i <= lod if lod < 50 else i > lod))}
        manifest = self.root / "frame_selection.json" if self.root else None
        self.selection_known = not self.root or self.config.get("frame_selection_stride", 1) == 1
        selected = None
        if manifest and manifest.is_file():
            selected = {Path(path).name for path in json.loads(manifest.read_text())["selected_images"]}
            self.selection_known = True
        for frame in self.frames:
            frame.split = "source" if self.root is None else "test" if frame.id in test_ids else (
                "train" if selected is None or Path(frame.name).name in selected else "unused")
        self.positions = np.stack([frame.center for frame in self.frames])
        cloud = self.xyz[::max(1, len(self.xyz) // 10000)] if len(self.xyz) else self.positions
        low, high = np.quantile(np.concatenate((cloud, self.positions)), [.02, .98], axis=0)
        self.center, self.radius = (low + high) / 2, max(float(np.linalg.norm(high - low) / 2), .001)
        up = -np.mean([frame.rotation[1] for frame in self.frames], axis=0)
        self.up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0., 0., 1.])
        counts = {}
        for frame in self.frames:
            counts[Path(frame.name).name] = counts.get(Path(frame.name).name, 0) + 1
        self.collisions = {name for name, count in counts.items() if count > 1}
        self.association_counts = {}
        self.analysis = None

    def image_path(self, frame):
        # Deliberately reproduce training's basename mapping, while reporting
        # discrepancies with COLMAP's declared relative name in the viewer.
        return self.images_directory / Path(frame.name).name

    def groups(self):
        groups = {"All COLMAP images": list(range(len(self.frames)))}
        for split, name in (("train", "Training subset"), ("test", "Test views"), ("unused", "Not selected")):
            values = [i for i, frame in enumerate(self.frames) if frame.split == split]
            if values:
                groups[name] = values
        return groups

    def frame_matches(self, index):
        """Read only the selected feature records; preserve exact int64 IDs."""
        xys, point_ids = read_observations(self.images_file, self.suffix, self.frames[index])
        locations = np.searchsorted(self.ids, point_ids)
        matched = np.zeros(len(point_ids), bool)
        if len(self.ids):
            matched = (point_ids >= 0) & (locations < len(self.ids))
            matched &= self.ids[np.minimum(locations, len(self.ids) - 1)] == point_ids
        self.association_counts[index] = int(matched.sum())
        return xys, point_ids, locations, matched

    def next_with_matches(self, index, indices):
        """Find another supported view without opening or decoding any RGB files."""
        start = indices.index(index) if index in indices else -1
        for step in range(1, len(indices) + 1):
            candidate = indices[(start + step) % len(indices)]
            if candidate == index:
                continue
            if candidate not in self.association_counts:
                if self.frames[candidate].count:
                    self.frame_matches(candidate)
                else:
                    self.association_counts[candidate] = 0
            if self.association_counts[candidate]:
                return candidate
        return None

    def cloud_overlay(self, pixels, frame, calibration, training, limit):
        """Coarse projection without observation matches or an occlusion test."""
        chosen = np.linspace(0, len(self.xyz) - 1, min(100000, len(self.xyz)), dtype=int)
        projected, camera_xyz = project_points(self.xyz[chosen], frame, calibration)
        width, height = pixels.shape[1], pixels.shape[0]
        if training:
            matrix = np.array(calibration["matrix"])
            with np.errstate(divide="ignore", invalid="ignore"):
                projected = camera_xyz[:, :2] / camera_xyz[:, 2:3]
                projected *= [matrix[0, 0] * width / calibration["width"],
                              matrix[1, 1] * height / calibration["height"]]
                projected += [(width - 1) / 2, (height - 1) / 2]
        else:
            projected = projected * [width / calibration["width"], height / calibration["height"]] - .5
        valid = (camera_xyz[:, 2] > 0) & np.isfinite(projected).all(axis=1)
        valid &= ((projected[:, 0] >= 0) & (projected[:, 0] < width) &
                  (projected[:, 1] >= 0) & (projected[:, 1] < height))
        visible = np.flatnonzero(valid)
        if len(visible) > limit:
            visible = visible[np.linspace(0, len(visible) - 1, limit, dtype=int)]
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        for x, y in projected[visible]:
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), outline=(255, 177, 49))
        return np.asarray(image), len(visible)

    def sparse_projection(self, index, resolution, space="Training input", max_points=250000):
        """Render sparse RGB landmarks into a separate image with a point depth buffer."""
        frame = self.frames[index]
        calibration = self.options[frame.camera_id]["colmap_calibration"]
        chosen = np.linspace(0, len(self.xyz) - 1, min(max_points, len(self.xyz)), dtype=int)
        projected, camera_xyz = project_points(self.xyz[chosen], frame, calibration)
        width, height = resolution
        if space == "Training input":
            matrix = np.array(calibration["matrix"])
            with np.errstate(divide="ignore", invalid="ignore"):
                projected = camera_xyz[:, :2] / camera_xyz[:, 2:3]
                projected *= [matrix[0, 0] * width / calibration["width"], matrix[1, 1] * height / calibration["height"]]
                projected += [(width - 1) / 2, (height - 1) / 2]
        else:
            projected = projected * [width / calibration["width"], height / calibration["height"]] - .5
        valid = (camera_xyz[:, 2] > 0) & np.isfinite(projected).all(axis=1)
        valid &= ((projected[:, 0] >= 0) & (projected[:, 0] <= width - 1) &
                  (projected[:, 1] >= 0) & (projected[:, 1] <= height - 1))
        indices = np.flatnonzero(valid)
        xy = np.rint(projected[indices]).astype(int)
        z = camera_xyz[indices, 2]
        order = np.argsort(z, kind="stable")
        _, first = np.unique(xy[order, 1] * width + xy[order, 0], return_index=True)
        selected = order[first]
        xy, z, colors = xy[selected], z[selected], self.rgb[chosen[indices[selected]]]
        image = np.zeros((height, width, 3), np.uint8)
        depth = np.full((height, width), np.inf)
        # Centers are unique. Each shifted set is also unique, so updates
        # safely resolve the nearest sampled landmark across the 3x3 footprints.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = xy[:, 0] + dx, xy[:, 1] + dy
                inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                candidates = np.flatnonzero(inside)
                keep = candidates[z[candidates] < depth[y[candidates], x[candidates]]]
                image[y[keep], x[keep]] = colors[keep]
                depth[y[keep], x[keep]] = z[keep]
        return image, dict(projected_points=len(selected), sampled_points=len(chosen),
                           covered_pixels=int(np.isfinite(depth).sum()))

    def inspect(self, index, space="Training input", limit=500, mode="Both"):
        from utils.image_loading import load_image_tensor, open_image

        frame = self.frames[index]
        options = self.options[frame.camera_id]
        calibration = options["colmap_calibration"]
        path = self.image_path(frame)
        with open_image(path) as source:
            source_size = source.size
        resolution = image_resolution(source_size, self.resolution)
        xys, point_ids, locations, matched = self.frame_matches(index)
        xyz = self.xyz[locations[matched]] if len(self.ids) else np.empty((0, 3))
        observed = xys[matched]
        projected, camera_xyz = project_points(xyz, frame, calibration)
        positive = (camera_xyz[:, 2] > 0) & np.isfinite(camera_xyz).all(axis=1)
        raw_errors = np.linalg.norm(projected[positive] - observed[positive], axis=1)
        scaling = np.array(resolution) / [calibration["width"], calibration["height"]]
        raw_observed = observed * scaling - .5
        raw_projected = projected * scaling - .5
        # Actual CUDA ndc2Pix(v,S) = ((v+1)*S-1)/2: image-center principal
        # point is (S-1)/2 in array coordinates, not S/2.
        matrix = np.array(calibration["matrix"])
        matrix[0] *= scaling[0]
        matrix[1] *= scaling[1]
        native = np.full_like(observed, np.nan)
        native[positive] = camera_xyz[positive, :2] / camera_xyz[positive, 2:3] * [matrix[0, 0], matrix[1, 1]]
        native += (np.array(resolution) - 1) / 2
        training_observed = raw_observed.copy()
        if self.undistorted and len(observed):
            target = matrix.copy()
            target[0, 2], target[1, 2] = resolution[0] / 2, resolution[1] / 2
            # Match the current OpenCV remap exactly, including its pixel-index
            # convention. This audits current behavior rather than silently
            # correcting calibration while diagnosing it.
            training_observed = undistort_points(raw_observed.reshape(-1, 1, 2), matrix,
                                  np.array(calibration["distortion"]), target).reshape(-1, 2)
        training_valid = positive & np.isfinite(training_observed).all(axis=1)
        training_valid &= ((training_observed[:, 0] >= 0) & (training_observed[:, 0] < resolution[0]) &
                           (training_observed[:, 1] >= 0) & (training_observed[:, 1] < resolution[1]))
        if self.undistorted and len(observed):
            _, _, valid_map = rectification_maps(calibration, resolution, resolution)
            supported = cv2.remap(valid_map.astype(np.float32),
                                  training_observed[:, 0].astype(np.float32).reshape(-1, 1),
                                  training_observed[:, 1].astype(np.float32).reshape(-1, 1),
                                  cv2.INTER_LINEAR).ravel()
            training_valid &= supported >= .999
        train_errors = np.linalg.norm(native[training_valid] - training_observed[training_valid], axis=1)
        training = space == "Training input"
        pixels, mask = load_image_tensor(path, resolution, options if training else None, return_mask=True)
        pixels = (pixels.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        if mask is not None:
            pixels[mask[0].numpy() < .5] = (35, 38, 44)
        overlay = overlay_image(pixels, (training_observed if training else raw_observed)[positive],
                                (native if training else raw_projected)[positive], limit, mode)
        cloud_points = 0
        if mode == "Cloud projection":
            overlay, cloud_points = self.cloud_overlay(pixels, frame, calibration, training, limit)
        correspondence_status = ""
        if not len(xyz):
            if not len(point_ids):
                correspondence_status = "This image has no stored 2D feature observations in the COLMAP images file."
            elif not (point_ids >= 0).any():
                correspondence_status = "This image has feature observations, but none are associated with a triangulated 3D point."
            else:
                correspondence_status = "This image references 3D point IDs absent from the loaded points3D model."
        warnings = []
        if Path(frame.name).name in self.collisions:
            warnings.append("Duplicate basenames: training can map different COLMAP images to the same RGB file.")
        declared = self.images_directory / frame.name
        if declared != path:
            warnings.append(f"COLMAP declares {frame.name}; training uses basename {path.name}.")
        if source_size[0] * calibration["height"] != source_size[1] * calibration["width"]:
            warnings.append("RGB and COLMAP dimensions have different aspect ratios; check cropping/rotation/resizing.")
        details = {"image": frame.name, "split": frame.split, "training_path": str(path),
                   "source_size": list(source_size), "colmap_size": [calibration["width"], calibration["height"]],
                   "training_size": list(resolution), "associated_points": len(xyz),
                   "feature_observations": len(point_ids),
                   "triangulated_observations": int((point_ids >= 0).sum()),
                   "correspondence_status": correspondence_status, "cloud_overlay_points": cloud_points,
                   "valid_training_points": int(training_valid.sum()),
                   "behind_camera": int((~positive).sum()),
                   "within_native_near_cull": int(((camera_xyz[:, 2] > 0) & (camera_xyz[:, 2] <= .2)).sum()),
                   "depth_median": float(np.median(camera_xyz[positive, 2])) if positive.any() else None,
                   "colmap_reprojection_px": _statistics(raw_errors),
                   "training_projection_px": _statistics(train_errors), "warnings": warnings}
        if index:
            previous = self.frames[index - 1]
            relative = frame.rotation @ previous.rotation.T
            details["previous_translation"] = float(np.linalg.norm(frame.center - previous.center))
            details["previous_rotation_degrees"] = math.degrees(math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        return overlay, pixels, xyz[positive], details

    def export_report(self, output, count=24, group="Training subset", indices=None, analysis=None, progress=None):
        from utils.image_loading import CorruptImageError

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        groups = self.groups()
        candidates = groups.get(group, groups["All COLMAP images"])
        if indices is None:
            indices = [candidates[item] for item in np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)]
        records, cards = [], []
        for number, index in enumerate(indices):
            if progress and (number % 4 == 0 or number == len(indices) - 1):
                progress(f"Rendering audit comparisons: {number + 1}/{len(indices)} · {self.frames[index].name}")
            try:
                raw, query, _, details = self.inspect(index, "Original image")
                trained, _, _, _ = self.inspect(index, "Training input")
                cloud, projection = self.sparse_projection(index, details["training_size"], "Original image")
            except (CorruptImageError, FileNotFoundError) as error:
                records.append({"image": self.frames[index].name, "error": str(error)})
                cards.append(f'<article><h2>{html.escape(self.frames[index].name)}</h2><p>{html.escape(str(error))}</p></article>')
                continue
            files = []
            for suffix, pixels in (("query", query), ("cloud", cloud), ("raw", raw), ("training", trained)):
                image = Image.fromarray(pixels)
                name = f"{number:03d}_{suffix}.png"
                image.save(output / name)
                files.append(name)
            records.append(details)
            status = (details["correspondence_status"] + " Numeric residuals are unavailable."
                      if details["associated_points"] == 0 else f"{details['associated_points']:,} associated points; cyan observations / orange projections / red residuals.")
            cards.append(f'<article><h2>{html.escape(details["image"])}</h2>'
                         f'<p><strong>{html.escape(status)}</strong></p>'
                         f'<p>{html.escape(json.dumps({key: details[key] for key in ("split", "colmap_reprojection_px", "training_projection_px", "warnings")}))}</p>'
                         f'<div><figure><figcaption>Captured RGB</figcaption><a href="{files[0]}"><img src="{files[0]}"></a></figure>'
                         f'<figure><figcaption>Reprojected sparse cloud · {projection["projected_points"]:,} points</figcaption><a href="{files[1]}"><img src="{files[1]}"></a></figure></div>'
                         f'<div><figure><figcaption>Original image + matched reprojection overlay</figcaption><a href="{files[2]}"><img src="{files[2]}"></a></figure>'
                         f'<figure><figcaption>Training input + native projection</figcaption><a href="{files[3]}"><img src="{files[3]}"></a></figure></div></article>')
        (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        overview = ""
        if analysis:
            from utils.pose_audit_analysis import summary_text
            overview = '<pre>' + html.escape(summary_text(analysis)) + '</pre>'
        page = ('<!doctype html><meta charset="utf-8"><title>Scaffold-GS pose audit</title>'
                '<style>body{background:#111820;color:#e5edf5;font:15px system-ui;margin:24px}h1{font-size:26px}'
                'article{border-top:1px solid #344353;padding:18px 0}article div{display:flex;gap:12px}'
                'figure{margin:0;flex:1;min-width:0}img{width:100%}p{overflow-wrap:anywhere;color:#bbc9d7}'
                'figcaption{margin:8px 0}@media(max-width:800px){article div{display:block}}</style>'
                '<h1>Image / pose audit</h1><p>Cyan circles: COLMAP observations. Orange crosses: projected 3D points. '
                'Red lines: residuals. Click an image to inspect pixels. Statistics use all associated points; overlays are sampled.</p>'
                '<p>The separate cloud image projects the sparse RGB landmarks using the saved camera. Black areas have no sampled points; '
                'nearest-point depth buffering is not a complete surface visibility test. This is not a trained Gaussian render.</p>'
                '<p>Low reprojection error measures COLMAP self-consistency. Inspect the marks on real image features: '
                'it does not prove that filenames still identify the same captured frames or that poses are globally correct.</p>'
                + overview + ''.join(cards))
        (output / "index.html").write_text(page)
        return output / "index.html"
