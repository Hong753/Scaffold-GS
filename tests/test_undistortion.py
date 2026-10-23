"""Check projection geometry, lazy cache behavior, masks and matched scoring."""

from argparse import Namespace
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from compare_undistortion import summarize, training_command
from utils.colmap_undistortion import (cache_path, calibration_summary, colmap_image_options,
                                      common_grid_psnr, rectification_maps, undistort_points)
from utils.graphics_utils import focal2fov
from utils.image_loading import CorruptImageError, load_image_tensor, sample_training_camera
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.undistortion_comparison import evaluation_cameras


ROOT = Path(__file__).resolve().parents[1]


def intrinsics(model="SIMPLE_RADIAL", params=(100., 60., 40., 0.12), width=120, height=80):
    return SimpleNamespace(id=1, model=model, params=np.array(params), width=width, height=height)


@pytest.mark.parametrize("model,params,expected", [
    ("SIMPLE_PINHOLE", [100, 60, 40], [0, 0, 0, 0]),
    ("PINHOLE", [100, 110, 60, 40], [0, 0, 0, 0]),
    ("SIMPLE_RADIAL", [100, 60, 40, .1], [.1, 0, 0, 0]),
    ("RADIAL", [100, 60, 40, .1, .02], [.1, .02, 0, 0]),
    ("OPENCV", [100, 110, 60, 40, .1, .02, .003, -.004], [.1, .02, .003, -.004]),
    ("FULL_OPENCV", [100, 110, 60, 40, .1, .02, .003, -.004, .01, .02, .03, .04],
     [.1, .02, .003, -.004, .01, .02, .03, .04]),
])
def test_colmap_parameter_order(model, params, expected):
    calibration = colmap_image_options(intrinsics(model, params))["colmap_calibration"]
    np.testing.assert_allclose(calibration["distortion"], expected)
    assert calibration["matrix"][0][0] == 100
    assert calibration["matrix"][1][1] == (110 if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"} else 100)


def test_unsupported_camera_fails_with_actionable_error():
    with pytest.raises(ValueError, match="image_undistorter"):
        colmap_image_options(intrinsics("OPENCV_FISHEYE"))


def test_remap_matches_colmap_perspective_projection_and_centers_principal_point():
    camera = intrinsics("OPENCV", [100, 105, 57, 42, .12, .02, .003, -.004])
    calibration = colmap_image_options(camera)["colmap_calibration"]
    x, y, valid = rectification_maps(calibration, (60, 40), (120, 80))
    for u, v in [(12, 10), (30, 20), (45, 28)]:
        point = np.array([[(u - 30) / 50, (v - 20) / 52.5, 1.]], dtype=np.float64)
        projected, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3),
                                         np.array(calibration["matrix"]), np.array(calibration["distortion"]))
        np.testing.assert_allclose([x[v, u], y[v, u]], projected.ravel(), atol=1e-5)
        assert valid[v, u]
    assert x[20, 30] == pytest.approx(57) and y[20, 30] == pytest.approx(42)


def make_image(tmp_path):
    path = tmp_path / "frame.png"
    pixels = np.random.default_rng(7).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    options = colmap_image_options(intrinsics(), True, tmp_path / "cache")
    return path, options


def test_cache_is_lossless_and_second_selection_does_not_read_source_or_remap(tmp_path, monkeypatch):
    path, options = make_image(tmp_path)
    first, mask = load_image_tensor(path, (60, 40), options, return_mask=True)
    destination = cache_path(path, (60, 40), options)
    with Image.open(destination) as cached:
        assert cached.mode == "RGBA" and cached.size == (60, 40)
    original_open = Image.open

    def guard_open(filename, *args, **kwargs):
        assert Path(filename) != path, "Cache hit decoded the source image"
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(Image, "open", guard_open)
    monkeypatch.setattr(cv2, "remap", lambda *a, **k: pytest.fail("Cache hit remapped an image"))
    second, second_mask = load_image_tensor(path, (60, 40), options, return_mask=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(mask, second_mask, rtol=0, atol=0)
    assert mask.min() == 0 and mask.max() == 1


def test_cache_keys_include_source_resolution_and_calibration(tmp_path):
    path, options = make_image(tmp_path)
    first = cache_path(path, (60, 40), options)
    assert first != cache_path(path, (120, 80), options)
    changed = colmap_image_options(intrinsics(params=(100, 60, 40, .2)), True, tmp_path / "cache")
    assert first != cache_path(path, (60, 40), changed)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    assert first != cache_path(path, (60, 40), options)


def test_corrupt_cache_is_rebuilt_but_corrupt_source_is_strictly_rejected(tmp_path):
    path, options = make_image(tmp_path)
    expected = load_image_tensor(path, (60, 40), options)
    destination = cache_path(path, (60, 40), options)
    destination.write_bytes(b"broken png")
    torch.testing.assert_close(load_image_tensor(path, (60, 40), options), expected)
    path.write_bytes(b"broken source")
    with pytest.raises(CorruptImageError):
        load_image_tensor(path, (60, 40), options)


def test_cache_permission_errors_are_not_misclassified_as_bad_images(tmp_path, monkeypatch):
    path, options = make_image(tmp_path)
    from utils import colmap_undistortion

    def denied(*args, **kwargs):
        raise PermissionError(13, "read-only cache")

    monkeypatch.setattr(colmap_undistortion, "_atomic_png", denied)
    with pytest.raises(PermissionError):
        load_image_tensor(path, (60, 40), options)


@pytest.fixture
def camera_utils(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    scene = ModuleType("scene")
    scene.__path__ = [str(ROOT / "scene")]
    monkeypatch.setitem(sys.modules, "scene", scene)
    for name, filename in [("scene.cameras", "scene/cameras.py"),
                           ("_undistort_camera_utils", "utils/camera_utils.py")]:
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def test_camera_setup_is_metadata_only_and_selected_frame_stays_uncached_in_ram(camera_utils, tmp_path, monkeypatch):
    path, options = make_image(tmp_path)
    info = SimpleNamespace(uid=0, R=np.eye(3), T=np.zeros(3), FovX=1., FovY=.8,
                           image_path=str(path), image_name="frame", image_options=options)
    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, "load", lambda *a, **k: pytest.fail("Camera setup decoded pixels"))
        cameras = [camera_utils.loadCam(Namespace(resolution=2, data_device="cpu"), i, info, 1.) for i in range(20)]
    assert not list((tmp_path / "cache").glob("**/*.png"))
    selected, image, stack, mask = sample_training_camera(cameras, None, "cpu", return_mask=True)
    assert image.shape == (3, 40, 60) and mask.shape == (1, 40, 60)
    assert all(camera._image is None for camera in cameras)
    assert all(not any(isinstance(value, Image.Image) for value in vars(camera).values()) for camera in cameras)
    assert len(list((tmp_path / "cache").glob("**/*.png"))) == 1


def test_masked_loss_psnr_and_ssim_ignore_invalid_border_and_have_zero_border_gradient():
    target = torch.full((3, 48, 64), .5)
    prediction = target.clone()
    prediction[:, :6] = 20
    prediction[:, 20:28, 25:35] += .1
    prediction.requires_grad_()
    mask = torch.ones(1, 48, 64)
    mask[:, :6] = 0
    loss = l1_loss(prediction, target, mask=mask) + 1 - ssim(prediction, target, mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad[:, :6].eq(0).all()
    assert prediction.grad[:, 22:26, 27:33].abs().sum() > 0
    squared = (prediction.detach()[:, 6:] - target[:, 6:]).square().mean(dim=(1, 2))
    torch.testing.assert_close(psnr(prediction.detach(), target, mask=mask).flatten(), -10 * squared.log10())
    perfect = target.clone()
    perfect[:, :6] = 20
    torch.testing.assert_close(ssim(perfect, target, mask=mask), torch.tensor(1.))


def test_all_valid_mask_preserves_original_losses_and_metrics():
    a, b = torch.rand(2, 3, 32, 40), torch.rand(2, 3, 32, 40)
    mask = torch.ones(2, 1, 32, 40)
    torch.testing.assert_close(l1_loss(a, b, mask=mask), l1_loss(a, b))
    torch.testing.assert_close(ssim(a, b, mask=mask), ssim(a, b))
    torch.testing.assert_close(psnr(a, b, mask=mask), psnr(a, b))


def test_common_grid_scores_both_runs_on_identical_original_pixels(tmp_path):
    camera = intrinsics(params=(70., 60., 40., .35))
    baseline = colmap_image_options(camera, False, tmp_path / "cache")
    corrected = colmap_image_options(camera, True, tmp_path / "cache")
    yy, xx = np.indices((80, 120), dtype=np.float32)
    points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    matrix = np.array(baseline["colmap_calibration"]["matrix"])
    ideal = undistort_points(points, matrix, np.array([.35, 0., 0., 0.]), matrix).reshape(80, 120, 2)
    color = lambda u, v: np.stack((u / 160 + .1, v / 120 + .1, (u + v) / 260 + .1), axis=-1)
    original = (color(ideal[..., 0], ideal[..., 1]) * 255).round().clip(0, 255).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(original).save(path)
    prediction = torch.from_numpy(color(xx, yy)).permute(2, 0, 1)
    view = SimpleNamespace(image_width=120, image_height=80, image_path=str(path), image_options=baseline)
    raw_score, raw_pixels = common_grid_psnr(prediction, view)
    view.image_options = corrected
    corrected_score, corrected_pixels = common_grid_psnr(prediction, view, tmp_path / "preview.png")
    assert corrected_pixels == raw_pixels and 0 < raw_pixels < 120 * 80
    assert corrected_score > raw_score + 10
    with Image.open(tmp_path / "preview.png") as preview:
        assert preview.size == (360, 80)


@pytest.mark.parametrize("resolution", [(120, 80), (60, 40), (45, 30)])
def test_zero_distortion_is_identity_and_reports_zero_displacement(tmp_path, resolution):
    path, _ = make_image(tmp_path)
    camera = intrinsics(params=(100, 60, 40, 0.))
    options = colmap_image_options(camera, True, tmp_path / "cache")
    rgb, mask = load_image_tensor(path, resolution, options, return_mask=True)
    torch.testing.assert_close(rgb, load_image_tensor(path, resolution), rtol=0, atol=0)
    assert mask.eq(1).all()
    summary = calibration_summary(options["colmap_calibration"])
    assert summary["sampled_max_shift_px"] == pytest.approx(0, abs=1e-10)
    assert summary["principal_point_shift_px"] == [0, 0]


def test_colmap_reader_preserves_pose_fov_and_retains_only_preprocessing_metadata(tmp_path):
    source = ast.parse((ROOT / "scene/dataset_readers.py").read_text())
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in {"CameraInfo", "readColmapCameras"}]
    namespace = dict(np=np, os=os, sys=sys, NamedTuple=NamedTuple, focal2fov=focal2fov,
                     qvec2rotmat=lambda q: np.eye(3))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<COLMAP reader>", "exec"), namespace)
    ext = SimpleNamespace(camera_id=1, qvec=np.array([1, 0, 0, 0]), tvec=[1, 2, 3], name="frame.png")
    plain = namespace["readColmapCameras"]({1: ext}, {1: intrinsics()}, str(tmp_path))[0]
    rectified = namespace["readColmapCameras"]({1: ext}, {1: intrinsics()}, str(tmp_path), True)[0]
    assert plain.image_options is None and rectified.image_options["undistort_colmap"]
    assert plain.FovX == rectified.FovX and plain.FovY == rectified.FovY
    np.testing.assert_array_equal(plain.R, rectified.R)
    np.testing.assert_array_equal(plain.T, rectified.T)
    assert rectified.image is None and rectified.image_path == str(tmp_path / "frame.png")


def test_evaluation_subset_is_stable_and_spans_full_sequence():
    cameras = [SimpleNamespace(image_name=f"{i:04d}", image_path=f"/{i}.png") for i in range(100)]
    subset = evaluation_cameras(list(reversed(cameras)), 5)
    assert subset == evaluation_cameras(cameras, 5)
    assert subset[0] is cameras[0] and subset[-1] is cameras[-1]


def test_launch_commands_differ_only_by_mode_and_output(tmp_path):
    args = Namespace(source=str(tmp_path), iterations=40000, gpu="0", resolution=-1,
                     appearance_dim=0, voxel_size=.001, frame_selection_stride=3,
                     eval_views=32, cache_dir=str(tmp_path / "cache"), port=6009)
    raw, corrected = [training_command(args, tmp_path, mode) for mode in (False, True)]
    assert corrected[-1] == "--undistort_images"
    raw[raw.index("-m") + 1] = corrected[corrected.index("-m") + 1]
    assert raw == corrected[:-1]


def test_comparison_summary_rejects_different_support(tmp_path):
    data = {"iteration": 7000, "split": "train", "native_psnr": 18., "raw_grid_psnr": 18.,
            "views": [{"image": "frame", "raw_grid_pixels": 100, "raw_grid_psnr": 18.}]}
    for name in ("baseline", "undistorted"):
        target = tmp_path / name / "comparison_metrics/007000_train.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(data))
    rows = summarize(tmp_path)
    assert rows[0]["delta_db"] == 0 and (tmp_path / "comparison.md").exists()
    data["views"][0]["raw_grid_pixels"] = 99
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="support differ"):
        summarize(tmp_path)


def test_training_report_writes_native_and_common_grid_metrics(camera_utils, tmp_path):
    path, options = make_image(tmp_path)
    info = SimpleNamespace(uid=0, R=np.eye(3), T=np.zeros(3), FovX=1., FovY=.8,
                           image_path=str(path), image_name="frame", image_options=options)
    camera = camera_utils.loadCam(Namespace(resolution=2, data_device="cpu"), 0, info, 1.)
    scene = SimpleNamespace(gaussians=torch.nn.Module(), model_path=str(tmp_path / "model"),
                            compare_undistortion=True, comparison_eval_views=32,
                            getTrainCameras=lambda: [camera], getTestCameras=lambda: [camera])
    source = ast.parse((ROOT / "train.py").read_text())
    node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "training_report")
    namespace = dict(torch=torch, Scene=object, psnr=psnr, prefilter_voxel=lambda *a: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "train.py", "exec"), namespace)
    messages = []
    logger = SimpleNamespace(info=messages.append, warning=lambda *a: pytest.fail(str(a)))
    namespace["training_report"](None, "scene", 7000, torch.tensor(0.), torch.tensor(0.),
                                  l1_loss, 0., [7000], scene,
                                  lambda *a, **k: {"render": torch.full((3, 40, 60), .5)},
                                  (None, torch.zeros(3)), logger=logger)
    for split in ("train", "test"):
        payload = json.loads((tmp_path / "model/comparison_metrics" / f"007000_{split}.json").read_text())
        assert payload["views"][0]["image"] == "frame"
        assert payload["views"][0]["raw_grid_pixels"] < 40 * 60
        assert np.isfinite(payload["native_psnr"]) and np.isfinite(payload["raw_grid_psnr"])
        assert (tmp_path / f"model/comparison_previews/7000/{split}/000.png").exists()
    assert scene.gaussians.training


@pytest.mark.parametrize("filename", ["train.py", "metrics.py"])
def test_offline_metrics_preserve_alpha_mask_and_ignore_invalid_pixels(tmp_path, filename):
    # Exercise the actual streaming evaluator; replace only LPIPS and logging.
    source = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in {"readImages", "evaluate"}]

    class Perceptual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, a, b):
            return (a - b).square().mean()

    def to_tensor(image):
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255

    namespace = dict(os=os, Path=Path, Image=Image, torch=torch, json=json,
                     tf=SimpleNamespace(to_tensor=to_tensor), tqdm=lambda it, **k: it,
                     ssim=ssim, psnr=psnr, lpips_fn=Perceptual())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    root = tmp_path / "model"
    renders, gts = root / "test/ours_1/renders", root / "test/ours_1/gt"
    renders.mkdir(parents=True)
    gts.mkdir()
    rgb = np.full((32, 48, 3), 128, dtype=np.uint8)
    predicted = np.full_like(rgb, 140)
    predicted[:, :10] = 255
    alpha = np.full((32, 48, 1), 255, dtype=np.uint8)
    alpha[:, :10] = 0
    Image.fromarray(predicted).save(renders / "00000.png")
    Image.fromarray(np.dstack((rgb, alpha))).save(gts / "00000.png")
    if filename == "train.py":
        namespace["evaluate"](str(root), visible_count=[1], logger=SimpleNamespace(
            info=lambda *a: None, warning=lambda *a: None))
    else:
        namespace["evaluate"]([str(root)])
    result = json.loads((root / "results.json").read_text())["ours_1"]
    assert result["PSNR"] == pytest.approx(-20 * np.log10(12 / 255), abs=1e-5)
