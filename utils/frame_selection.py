"""Pick a fixed, sharp training frame from each consecutive temporal group."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import warnings

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm


_VERSION = 1


def _path(camera):
    return str(Path(camera.image_path).resolve())


def _sequence_key(camera):
    # frame2 precedes frame10 even when names are not zero-padded.
    parts = re.split(r"(\d+)", Path(camera.image_path).name.casefold())
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts), _path(camera)


def _score_frame(item):
    path, max_size = item
    import cv2

    try:
        with Image.open(path) as source:
            gray = source.convert("L")  # Decode fully, so truncated files fail.
            if max_size:
                gray.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(gray)
            score = float(cv2.Laplacian(pixels, cv2.CV_64F, ksize=1).var())
        return score, None
    except OSError as error:
        if error.errno is not None:
            raise OSError(error.errno, f"Could not score image: {error}", path) from error
        return None, str(error)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _restore_selection(cameras, payload):
    by_path = {_path(camera): camera for camera in cameras}
    selected = payload["selected_images"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Frame selection is empty or contains duplicate paths")
    missing = [path for path in selected if path not in by_path]
    if missing:
        raise ValueError(f"Saved training frame is absent from this dataset/split: {missing[0]}")
    return [by_path[path] for path in selected]


def select_training_frames(cameras, stride=1, max_size=1024, workers=4,
                           cache_dir="outputs/frame_selection", manifest_path=None,
                           reuse_manifest=False):
    """Select before camera construction; retain paths/scores, never image pixels.

    Selection runs after the dataset's train/test split. A model's saved
    manifest takes precedence on reload so appearance embedding IDs stay stable.
    """
    if stride < 1 or workers < 1 or max_size < 0:
        raise ValueError("frame_selection_stride/workers must be positive; max_size must be >= 0")

    if reuse_manifest and manifest_path and Path(manifest_path).is_file():
        with open(manifest_path) as stream:
            payload = json.load(stream)
        selected = _restore_selection(cameras, payload)
        print(f"[Frame selection] Loaded {len(selected)}/{len(cameras)} saved training frames")
        return selected

    if stride == 1:
        return list(cameras)

    ordered = sorted(cameras, key=_sequence_key)
    if not ordered:
        raise RuntimeError("No training frames available for sharpness selection")
    paths = [_path(camera) for camera in ordered]
    signatures = []
    for path in paths:
        stat = os.stat(path)
        signatures.append([path, stat.st_size, stat.st_mtime_ns])
    fingerprint = hashlib.sha256(json.dumps(
        [_VERSION, stride, max_size, signatures], separators=(",", ":")
    ).encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{fingerprint}.json"
    payload = None
    try:
        with cache_path.open() as stream:
            cached = json.load(stream)
        if (cached["fingerprint"] == fingerprint and cached["version"] == _VERSION
                and [frame["path"] for frame in cached["frames"]] == paths
                and all(frame["score"] is None or (
                    isinstance(frame["score"], (int, float)) and math.isfinite(frame["score"]))
                    for frame in cached["frames"])):
            payload = cached
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass

    if payload is None:
        # Only this bounded scoring pass holds decoded images in CPU memory.
        # All workers run with strict decoding; restore Pillow's global setting
        # before returning to the ordinary on-demand camera loader.
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(tqdm(executor.map(_score_frame, ((path, max_size) for path in paths)),
                                    total=len(paths), desc="Selecting sharp training frames"))
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        payload = {
            "version": _VERSION, "metric": "variance_of_laplacian", "stride": stride,
            "max_size": max_size, "fingerprint": fingerprint,
            "frames": [{"path": path, "score": score, "error": error}
                       for path, (score, error) in zip(paths, results)],
        }
    else:
        print("[Frame selection] Using cached sharpness scores")

    frames = payload["frames"]
    selected_indices = []
    for start in range(0, len(frames), stride):
        valid = [index for index in range(start, min(start + stride, len(frames)))
                 if frames[index]["score"] is not None]
        if valid:
            # Ties keep the first frame in temporal order.
            selected_indices.append(max(valid, key=lambda index: frames[index]["score"]))
    for frame in frames:
        if frame["score"] is None:
            warnings.warn(f"Skipping corrupt image {frame['path']!r}: {frame['error']}",
                          RuntimeWarning, stacklevel=2)
    if not selected_indices:
        raise RuntimeError("No decodable training images remain after sharpness selection")
    payload["selected_images"] = [paths[index] for index in selected_indices]
    _write_json(cache_path, payload)
    if manifest_path:
        _write_json(manifest_path, payload)
    print(f"[Frame selection] Keeping {len(selected_indices)}/{len(ordered)} training frames "
          f"(sharpest per {stride}, max side {max_size or 'native'} px)")
    return [ordered[index] for index in selected_indices]
