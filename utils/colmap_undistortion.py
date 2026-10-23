"""Lazy COLMAP rectification, lossless disk caching, and common-grid evaluation."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image


_CACHE_VERSION = 1
_COUNTS = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
           "RADIAL": 5, "OPENCV": 8, "FULL_OPENCV": 12}


def undistort_points(points, matrix, distortion, target):
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-8)
    if hasattr(cv2, "undistortPointsIter"):
        return cv2.undistortPointsIter(points, matrix, distortion, None, target, criteria)
    # OpenCV 5 merged the iterative overload into undistortPoints.
    return cv2.undistortPoints(points, matrix, distortion, P=target, criteria=criteria)


def colmap_image_options(camera, undistort=False, cache_dir="outputs/undistorted"):
    """Keep calibration metadata on cameras; never decode images here.

    Output intrinsics retain fx/fy and use the image center, as required by the
    native renderer. Invalid borders are masked rather than changing the FOV.
    """
    model, p = camera.model, np.asarray(camera.params, dtype=np.float64)
    if model not in _COUNTS:
        raise ValueError(f"On-demand undistortion does not support COLMAP {model}; "
                         "use COLMAP image_undistorter to produce pinhole images first.")
    if len(p) != _COUNTS[model] or not np.isfinite(p).all():
        raise ValueError(f"Invalid parameters for COLMAP {model}: {p.tolist()}")
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
        distortion = [p[3] if len(p) > 3 else 0., p[4] if len(p) > 4 else 0., 0., 0.]
    else:
        fx, fy, cx, cy = p[:4]
        distortion = p[4:].tolist() if len(p) > 4 else [0., 0., 0., 0.]
    if min(fx, fy, camera.width, camera.height) <= 0:
        raise ValueError("Camera focal lengths and dimensions must be positive")
    return {"colmap_calibration": {
        "model": model, "width": int(camera.width), "height": int(camera.height),
        "matrix": [[float(fx), 0., float(cx)], [0., float(fy), float(cy)], [0., 0., 1.]],
        "distortion": list(map(float, distortion)),
    }, "undistort_colmap": bool(undistort), "undistort_cache_dir": str(cache_dir)}


def _matrices(calibration, resolution, source_size):
    source = np.array(calibration["matrix"], dtype=np.float64)
    target = source.copy()
    source[0] *= source_size[0] / calibration["width"]
    source[1] *= source_size[1] / calibration["height"]
    target[0] *= resolution[0] / calibration["width"]
    target[1] *= resolution[1] / calibration["height"]
    target[0, 2], target[1, 2] = resolution[0] / 2, resolution[1] / 2
    return source, target


@lru_cache(maxsize=2)
def _maps(calibration_json, resolution, source_size):
    # Only two calibration maps are retained, never decoded frame pixels.
    calibration = json.loads(calibration_json)
    source, target = _matrices(calibration, resolution, source_size)
    x, y = cv2.initUndistortRectifyMap(source, np.array(calibration["distortion"]),
                                     None, target, resolution, cv2.CV_32FC1)
    valid = ((x >= -1e-4) & (x <= source_size[0] - 1 + 1e-4) &
             (y >= -1e-4) & (y <= source_size[1] - 1 + 1e-4))
    return x, y, valid


def rectification_maps(calibration, resolution, source_size):
    return _maps(json.dumps(calibration, sort_keys=True), tuple(resolution), tuple(source_size))


def cache_path(path, resolution, options):
    stat = Path(path).stat()
    signature = {"version": _CACHE_VERSION, "opencv": cv2.__version__,
                 "path": str(Path(path).resolve()), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns, "resolution": list(resolution),
                 "calibration": options["colmap_calibration"]}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return Path(options["undistort_cache_dir"]) / digest[:2] / (digest + ".png")


def _atomic_png(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
            temporary = stream.name
            Image.fromarray(pixels).save(stream, format="PNG", compress_level=1)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_rectified_image(path, resolution, options, open_image):
    """Decode/remap only a selected frame; cache final-resolution RGBA PNG."""
    destination = cache_path(path, resolution, options)
    if destination.is_file():
        try:
            with open_image(destination) as cached:
                if cached.mode != "RGBA" or cached.size != tuple(resolution):
                    raise ValueError("Invalid rectification cache dimensions/format")
                return np.array(cached, copy=True)
        except (ValueError, OSError) as error:
            # A broken cache is reproducible. Source and filesystem errors
            # remain strict; permissions/disk errors must not silently skip data.
            if isinstance(error, OSError) and error.errno is not None:
                raise
            destination.unlink(missing_ok=True)
    with open_image(path) as source:
        # Match baseline PIL resizing before applying the optional warp. Thus
        # zero distortion + centered principal point is exactly the baseline,
        # including antialiasing when source video is larger than training RGB.
        pixels = np.asarray(source.convert("RGB").resize(resolution))
        x, y, valid = rectification_maps(options["colmap_calibration"], resolution, resolution)
        rgb = cv2.remap(pixels, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgba = np.dstack((rgb, valid.astype(np.uint8) * 255))
    if not valid.any():
        raise ValueError(f"Undistortion leaves no valid pixels: {path}")
    _atomic_png(destination, rgba)
    return rgba


@lru_cache(maxsize=2)
def _raw_grid_maps(calibration_json, resolution):
    calibration = json.loads(calibration_json)
    raw, target = _matrices(calibration, resolution, resolution)
    # COLMAP's raw principal point, unlike the centered target camera.
    yy, xx = np.indices((resolution[1], resolution[0]), dtype=np.float32)
    points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    mapped = undistort_points(points, raw, np.array(calibration["distortion"]), target).reshape(
        resolution[1], resolution[0], 2)
    x, y = np.ascontiguousarray(mapped[..., 0]), np.ascontiguousarray(mapped[..., 1])
    _, _, valid_output = rectification_maps(calibration, resolution, resolution)
    valid = ((x >= -1e-4) & (x <= resolution[0] - 1 + 1e-4) &
             (y >= -1e-4) & (y <= resolution[1] - 1 + 1e-4))
    # Use exactly the same support for baseline and rectified runs, including
    # interpolation footprints at the boundary of the rectified image.
    valid &= cv2.remap(valid_output.astype(np.float32), x, y, cv2.INTER_LINEAR) >= 0.999
    return x, y, valid


def common_grid_psnr(rendering, camera, preview_path=None):
    """Score both modes against original RGB on identical original-grid pixels.

    Rectified predictions are warped back, without an 8-bit round trip. This
    avoids treating smoother rectified targets or black borders as improvements.
    The image-level score averages RGB channel PSNR, like utils.image_utils.
    """
    from utils.image_loading import load_image_tensor

    options = camera.image_options
    calibration = options.get("colmap_calibration")
    if calibration is None:
        raise ValueError("Common-grid evaluation requires COLMAP calibration metadata")
    resolution = (camera.image_width, camera.image_height)
    x, y, valid = _raw_grid_maps(json.dumps(calibration, sort_keys=True), resolution)
    if not valid.any():
        raise ValueError(f"No shared raw-grid pixels for {camera.image_path}")
    predicted = rendering.detach().float().cpu().permute(1, 2, 0).numpy()
    if options.get("undistort_colmap"):
        predicted = cv2.remap(predicted, x, y, cv2.INTER_LINEAR)
    original = load_image_tensor(camera.image_path, resolution).permute(1, 2, 0).numpy()
    mse = ((predicted[valid].astype(np.float64) - original[valid]) ** 2).mean(axis=0)
    if preview_path is not None:
        error = np.abs(predicted - original) * 4
        panels = [np.where(valid[..., None], panel, 0) for panel in (original, predicted, error)]
        pixels = (np.concatenate(panels, axis=1).clip(0, 1) * 255).round().astype(np.uint8)
        _atomic_png(Path(preview_path), pixels)
    return float((-10 * np.log10(np.maximum(mse, 1e-12))).mean()), int(valid.sum())


def calibration_summary(calibration):
    """Estimate distortion-only displacement on a 17x13 pixel grid."""
    width, height = calibration["width"], calibration["height"]
    xx, yy = np.meshgrid(np.linspace(0, width - 1, 17), np.linspace(0, height - 1, 13))
    points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2).astype(np.float64)
    matrix = np.array(calibration["matrix"])
    ideal = undistort_points(points, matrix, np.array(calibration["distortion"]), matrix)
    shifts = np.linalg.norm(ideal - points, axis=-1)
    return {**calibration, "sampled_max_shift_px": float(shifts.max()),
            "sampled_median_shift_px": float(np.median(shifts)),
            "principal_point_shift_px": [float(width / 2 - matrix[0, 2]),
                                           float(height / 2 - matrix[1, 2])]}
