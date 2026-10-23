"""Load one camera image from disk without retaining a decoded-image cache."""

from contextlib import contextmanager
from random import randrange
from threading import Lock
import warnings

import numpy as np
from PIL import Image, ImageFile

from utils.general_utils import PILtoTorch


_decode_lock = Lock()
_warned_corrupt_paths = set()


class CorruptImageError(OSError):
    """Image data could not be decoded; filesystem failures remain separate."""

    def __init__(self, path, error):
        self.image_path = str(path)
        super().__init__(f"Could not decode camera image {self.image_path!r}: {error}")


def warn_corrupt_image(error):
    if error.image_path not in _warned_corrupt_paths:
        warnings.warn(f"Skipping corrupt image: {error}", RuntimeWarning, stacklevel=2)
        _warned_corrupt_paths.add(error.image_path)


@contextmanager
def open_image(path):
    """Read strictly, close the file, and distinguish corruption from I/O errors."""
    try:
        with _decode_lock:
            # Enforce strict decoding even if another module enabled recovery.
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
                with Image.open(path) as image:
                    yield image
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
    except OSError as error:
        if error.errno is None:
            raise CorruptImageError(path, error) from error
        raise OSError(error.errno, f"Could not load camera image: {error}", str(path)) from error


def sample_training_camera(cameras, stack, device, return_mask=False):
    """Return a valid view for this step, excluding known corrupt views."""
    while True:
        if not stack:
            stack = [camera for camera in cameras if not camera.image_invalid]
            if not stack:
                raise RuntimeError("No decodable training images remain; check the reported image paths.")
        camera = stack.pop(randrange(len(stack)))
        image = camera.try_load_image(device, return_mask=True) if return_mask else camera.try_load_image(device)
        if image is not None:
            if return_mask:
                rgb, mask = image
                return camera, rgb, stack, mask
            return camera, image, stack


def load_image_tensor(path, resolution, options=None, return_mask=False):
    """Decode, preprocess and resize one image; always close its source file."""
    options = options or {}
    if options.get("undistort_colmap"):
        from utils.colmap_undistortion import load_rectified_image
        import torch

        pixels = load_rectified_image(path, resolution, options, open_image)
        rgb = torch.from_numpy(pixels[:, :, :3].copy()).permute(2, 0, 1).float() / 255.
        mask = torch.from_numpy(pixels[:, :, 3:4].copy()).permute(2, 0, 1).float() / 255.
        return (rgb.contiguous(), mask.contiguous()) if return_mask else rgb.contiguous()
    with open_image(path) as source:
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
    return (rgb.contiguous(), None) if return_mask else rgb.contiguous()
