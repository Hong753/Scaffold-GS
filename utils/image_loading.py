"""Load one camera image from disk without retaining a decoded-image cache."""

import numpy as np
from PIL import Image

from utils.general_utils import PILtoTorch


def load_image_tensor(path, resolution, options=None):
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
