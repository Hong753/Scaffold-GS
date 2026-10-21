"""Load one camera image from disk without retaining a decoded-image cache."""

from threading import Lock
import warnings

import numpy as np
from PIL import Image, ImageFile

from utils.general_utils import PILtoTorch


_decode_lock = Lock()
_warned_truncated_paths = set()


def load_image_tensor(path, resolution, options=None):
    """Load one image, retrying recoverable truncation without caching pixels."""
    try:
        # Pillow's recovery setting is global. Serialize our decodes so another
        # camera cannot observe the temporary setting, and always restore it.
        with _decode_lock:
            try:
                return _decode_image_tensor(path, resolution, options)
            except OSError as error:
                if "truncated" not in str(error).lower():
                    raise

            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                image = _decode_image_tensor(path, resolution, options)
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous

            key = str(path)
            if key not in _warned_truncated_paths:
                warnings.warn(
                    f"Recovered truncated image {key!r}. Some pixels may be "
                    "missing or corrupted; replace or re-export this source image.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_truncated_paths.add(key)
            return image
    except OSError as error:
        raise OSError(f"Could not load camera image {str(path)!r}: {error}") from error


def _decode_image_tensor(path, resolution, options=None):
    """Decode, preprocess and resize one image; always close its source file."""
    options = options or {}
    with Image.open(path) as source:
        image = source
        if "undistortion" in options:
            import cv2

            calibration = options["undistortion"]
            matrix = np.asarray(calibration["matrix"], dtype=np.float32)
            distortion = np.asarray(calibration["distortion"], dtype=np.float32)
            pixels = np.asarray(source.convert("RGB")) / 255.0
            pixels = cv2.undistort(pixels, matrix, distortion, None, matrix)
            image = Image.fromarray(np.asarray(pixels * 255.0, dtype=np.uint8), "RGB")
        elif "background" in options:
            pixels = np.asarray(source.convert("RGBA")) / 255.0
            background = np.asarray(options["background"])
            pixels = pixels[:, :, :3] * pixels[:, :, 3:4] + background * (1 - pixels[:, :, 3:4])
            image = Image.fromarray(np.asarray(pixels * 255.0, dtype=np.uint8), "RGB")
        tensor = PILtoTorch(image, resolution)

    rgb = tensor[:3].clamp(0.0, 1.0)
    if tensor.shape[0] == 4:
        rgb = rgb * tensor[3:4]
    return rgb.contiguous()
