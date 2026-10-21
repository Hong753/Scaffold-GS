import numpy as np
import pytest
import torch

from utils.voxel_utils import voxelize_points


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_grid_centers_match_numpy(dtype, device):
    rng = np.random.default_rng(8)
    points = rng.uniform(-4, 4, (5000, 3)).astype(dtype)
    points = np.concatenate((points, points[:1000], np.zeros((5, 3), dtype=dtype)))
    original = points.copy()
    for voxel_size in [0.001, 0.03, 0.5, 4.0]:
        expected = np.unique(np.round(points / voxel_size), axis=0) * voxel_size
        actual = voxelize_points(points, voxel_size, device)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_half_voxels_empty_and_degenerate_axes(dtype):
    points = np.array([[x, 0, -x] for x in [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]], dtype=dtype)
    np.testing.assert_array_equal(voxelize_points(points, 1.0), np.unique(np.round(points), axis=0))
    empty = voxelize_points(np.empty((0, 3), dtype=dtype), 0.001)
    assert empty.shape == (0, 3) and empty.dtype == dtype
    duplicate = np.repeat(points[:1], 100, axis=0)
    np.testing.assert_array_equal(voxelize_points(duplicate, 1.0), np.unique(np.round(duplicate), axis=0))


@pytest.mark.parametrize("points", [
    [[-1e10, -1e10, -1e10], [1e10, 1e10, 1e10]],
    [[1e20, 0, 0], [1e20, 0, 0], [-1e20, 0, 0]],
])
def test_oversized_keys_use_exact_fallback(points):
    points = np.asarray(points, dtype=np.float64)
    expected = np.unique(np.round(points / 0.001), axis=0) * 0.001
    np.testing.assert_array_equal(voxelize_points(points, 0.001), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_deduplication_matches_numpy():
    points = np.random.default_rng(9).uniform(-10, 10, (200000, 3)).astype(np.float32)
    expected = np.unique(np.round(points / 0.001), axis=0) * 0.001
    np.testing.assert_array_equal(voxelize_points(points, 0.001, "cuda"), expected)
