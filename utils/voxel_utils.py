"""Voxel-grid sampling using collision-free scalar keys.

Keep NumPy's original rounding and lexicographic anchor order, while replacing
the expensive three-column unique operation with one-dimensional deduplication.
"""

import math

import numpy as np


def voxelize_points(points, voxel_size, device=None):
    """Return np.unique(np.round(points / size), axis=0) * size.

    A requested torch device deduplicates the packed keys with torch.unique;
    None uses NumPy. Oversized coordinate ranges use the original row-wise
    operation. CUDA allocation failures fall back to packed keys on the CPU.
    Input points are never shuffled or modified.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    # Quantize on the CPU to preserve the original NumPy rounding, including
    # float32 half-voxel cases. Only deduplication is moved to the GPU.
    quantized = np.round(points / voxel_size)
    dtype = quantized.dtype
    if not len(quantized):
        return quantized * voxel_size
    if not np.isfinite(quantized).all():
        return np.unique(quantized, axis=0) * voxel_size

    lower = [int(value) for value in quantized.min(axis=0)]
    upper = [int(value) for value in quantized.max(axis=0)]
    bounds = np.iinfo(np.int64)
    dimensions = [high - low + 1 for low, high in zip(lower, upper)]
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    if min(lower) < bounds.min or max(upper) > bounds.max or volume > bounds.max:
        return np.unique(quantized, axis=0) * voxel_size

    # x-major mixed-radix keys preserve NumPy's x/y/z lexicographic order.
    # Python-integer bounds above guarantee that no int64 operation overflows.
    keys = quantized[:, 0].astype(np.int64) - lower[0]
    keys *= dimensions[1]
    keys += quantized[:, 1].astype(np.int64) - lower[1]
    keys *= dimensions[2]
    keys += quantized[:, 2].astype(np.int64) - lower[2]
    del quantized

    if device is None:
        unique_keys = np.unique(keys)
    else:
        import torch

        device = torch.device(device)
        tensor = None
        try:
            tensor = torch.from_numpy(keys).to(device=device)
            unique_keys = torch.unique(tensor, sorted=True).cpu().numpy()
        except RuntimeError as error:
            if device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            del tensor
            torch.cuda.empty_cache()
            print("[Init] CUDA voxel deduplication ran out of memory; using CPU packed keys.", flush=True)
            unique_keys = np.unique(keys)

    # Reconstruct centers, rather than averaging points or retaining an
    # arbitrary representative, so the Scaffold-GS initialization is preserved.
    centers = np.empty((len(unique_keys), 3), dtype=dtype)
    centers[:, 2] = unique_keys % dimensions[2] + lower[2]
    reduced = unique_keys // dimensions[2]
    centers[:, 1] = reduced % dimensions[1] + lower[1]
    centers[:, 0] = reduced // dimensions[1] + lower[0]
    centers *= voxel_size
    return centers
