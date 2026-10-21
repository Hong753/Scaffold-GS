"""Check disk-backed camera images without requiring the CUDA rasterizer."""

import ast
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple
import warnings

import numpy as np
import pytest
import torch
from PIL import Image, ImageFile

from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_loading import (CorruptImageError, load_image_tensor, open_image,
                                 sample_training_camera, warn_corrupt_image)
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def camera_utils(monkeypatch):
    # Only the existing camera-matrix CUDA transfers are redirected for CPU
    # testing. Production image decoding, resizing and device selection run.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    scene = ModuleType("scene")
    scene.__path__ = [str(ROOT / "scene")]
    monkeypatch.setitem(sys.modules, "scene", scene)
    for name, relative in [("scene.cameras", "scene/cameras.py"),
                           ("_disk_camera_utils", "utils/camera_utils.py")]:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def camera_info(path, options=None):
    return SimpleNamespace(uid=0, R=np.eye(3), T=np.zeros(3), FovX=0.8, FovY=0.6,
                           image=None, image_path=str(path), image_name=path.stem,
                           image_options=options)


def test_camera_creation_does_not_decode_or_cache_pixels(camera_utils, monkeypatch, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (24, 16), (80, 120, 160)).save(path)
    args = SimpleNamespace(resolution=2, data_device="cpu")

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Camera construction decoded image pixels")

    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, "load", unexpected_decode)
        cameras = [camera_utils.loadCam(args, index, camera_info(path), 1.0) for index in range(100)]
    assert all(camera._image is None for camera in cameras)
    assert all("original_image" not in vars(camera) for camera in cameras)
    assert all((camera.image_width, camera.image_height) == (12, 8) for camera in cameras)
    selected = cameras[17].load_image("cpu")
    with Image.open(path) as source:
        expected = PILtoTorch(source, (12, 8))
    torch.testing.assert_close(selected, expected)
    assert all(camera._image is None for camera in cameras)


def test_selected_camera_rereads_disk_and_keeps_no_tensor(camera_utils, tmp_path):
    path = tmp_path / "changing.png"
    Image.new("RGB", (16, 12), (255, 0, 0)).save(path)
    args = SimpleNamespace(resolution=1, data_device="cpu")
    camera = camera_utils.loadCam(args, 0, camera_info(path), 1.0)
    first = camera.load_image("cpu")
    Image.new("RGB", (16, 12), (0, 255, 0)).save(path)
    second = camera.load_image("cpu")
    assert first[0].eq(1).all() and second[1].eq(1).all()
    assert first[1].eq(0).all() and second[0].eq(0).all()
    assert camera._image is None
    torch.testing.assert_close(camera.original_image, second)


def jpeg_bytes():
    pixels = np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


@pytest.mark.parametrize("recovery_setting", [False, True])
def test_truncated_selected_image_is_strictly_rejected(monkeypatch, tmp_path, recovery_setting):
    path = tmp_path / "truncated.jpg"
    path.write_bytes(jpeg_bytes()[:-32])
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", recovery_setting)
    opened = []
    original_open = Image.open

    def track_open(*args, **kwargs):
        image = original_open(*args, **kwargs)
        opened.append(image)
        return image

    monkeypatch.setattr(Image, "open", track_open)
    with pytest.raises(CorruptImageError, match="truncated") as failure:
        load_image_tensor(path, (32, 24))
    assert str(path) in str(failure.value)
    assert opened and all(image.fp is None for image in opened)
    assert ImageFile.LOAD_TRUNCATED_IMAGES == recovery_setting


def test_corrupt_camera_is_skipped_once(camera_utils, monkeypatch, tmp_path):
    path = tmp_path / "truncated.jpg"
    path.write_bytes(jpeg_bytes()[:-32])
    camera = camera_utils.loadCam(SimpleNamespace(resolution=1, data_device="cpu"),
                                  0, camera_info(path), 1.0)
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image") as messages:
        assert camera.try_load_image("cpu") is None
    assert str(path) in str(messages[0].message)
    assert camera.image_invalid and camera._image is None

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Known corrupt image was decoded again")

    monkeypatch.setattr(Image.Image, "load", unexpected_decode)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        assert camera.try_load_image("cpu") is None
    assert not messages


def test_training_sampler_replaces_corrupt_view_within_step(camera_utils, monkeypatch, tmp_path):
    bad_path, good_path = tmp_path / "bad.jpg", tmp_path / "good.png"
    bad_path.write_bytes(jpeg_bytes()[:-32])
    Image.new("RGB", (64, 48), (0, 255, 0)).save(good_path)
    args = SimpleNamespace(resolution=1, data_device="cpu")
    cameras = [camera_utils.loadCam(args, index, camera_info(path), 1.0)
               for index, path in enumerate([bad_path, good_path])]
    monkeypatch.setattr("utils.image_loading.randrange", lambda size: 0)
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image"):
        selected, image, stack = sample_training_camera(cameras, None, "cpu")
    assert selected is cameras[1]
    assert image[1].eq(1).all() and image[[0, 2]].eq(0).all()
    assert cameras[0].image_invalid
    for _ in range(3):
        selected, _, stack = sample_training_camera(cameras, stack, "cpu")
        assert selected is cameras[1]
    assert all(camera._image is None for camera in cameras)


def test_training_stops_when_all_views_are_corrupt(camera_utils, tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(jpeg_bytes()[:-32])
    camera = camera_utils.loadCam(SimpleNamespace(resolution=1, data_device="cpu"),
                                  0, camera_info(path), 1.0)
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image"), \
            pytest.raises(RuntimeError, match="No decodable training images"):
        sample_training_camera([camera], None, "cpu")


def test_corrupt_header_is_skipped_without_decoding_other_views(camera_utils, monkeypatch, tmp_path):
    bad_path, good_path = tmp_path / "bad.jpg", tmp_path / "good.png"
    bad_path.write_bytes(jpeg_bytes()[:100])
    Image.new("RGB", (64, 48)).save(good_path)

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Camera construction decoded image pixels")

    monkeypatch.setattr(Image.Image, "load", unexpected_decode)
    args = SimpleNamespace(resolution=1, data_device="cpu")
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image"):
        cameras = camera_utils.cameraList_from_camInfos(
            [camera_info(bad_path), camera_info(good_path)], 1.0, args)
    assert len(cameras) == 1 and cameras[0].image_path == str(good_path)
    assert cameras[0]._image is None


@pytest.mark.parametrize("data", [b"not an image", jpeg_bytes()[:100]])
def test_unrecoverable_image_reports_path_and_restores_pillow(monkeypatch, tmp_path, data):
    path = tmp_path / "broken.jpg"
    path.write_bytes(data)
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)
    with pytest.raises(CorruptImageError, match="Could not decode camera image") as failure:
        load_image_tensor(path, (32, 24))
    assert str(path) in str(failure.value)
    assert isinstance(failure.value.__cause__, OSError)
    assert not ImageFile.LOAD_TRUNCATED_IMAGES


def test_filesystem_errors_are_not_classified_as_corruption(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_tensor(tmp_path / "missing.png", (32, 24))

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Image, "open", denied)
    with pytest.raises(PermissionError) as failure:
        load_image_tensor(tmp_path / "unreadable.png", (32, 24))
    assert not isinstance(failure.value, CorruptImageError)
    assert "unreadable.png" in str(failure.value)


@pytest.mark.parametrize("background", [None, [0, 0, 0], [1, 1, 1]])
def test_alpha_and_synthetic_backgrounds(tmp_path, background):
    path = tmp_path / "rgba.png"
    rgba = np.full((8, 12, 4), [200, 100, 50, 128], dtype=np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    options = {} if background is None else {"background": background}
    actual = load_image_tensor(path, (12, 8), options)
    normalized = rgba.astype(np.float64) / 255.0
    expected = normalized[:, :, :3] * normalized[:, :, 3:4]
    if background is not None:
        expected += np.asarray(background) * (1 - normalized[:, :, 3:4])
        expected = (expected * 255).astype(np.uint8).astype(np.float32) / 255
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32).permute(2, 0, 1))


def test_image_file_closes_on_decode_failure(monkeypatch, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 12)).save(path)
    opened = []
    original_open = Image.open

    def track_open(*args, **kwargs):
        image = original_open(*args, **kwargs)
        opened.append(image)
        return image

    def fail(*args, **kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(Image, "open", track_open)
    monkeypatch.setattr("utils.image_loading.PILtoTorch", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        load_image_tensor(path, (16, 12))
    assert opened and all(image.fp is None for image in opened)


class QuietProgress:
    def __init__(self, values, **kwargs):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def source_function(filename, name, namespace):
    source = ast.parse((ROOT / filename).read_text())
    node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("all_test_views_bad", [False, True])
def test_validation_uses_only_valid_views(camera_utils, tmp_path, all_test_views_bad):
    bad_path, good_path = tmp_path / "bad.jpg", tmp_path / "good.png"
    bad_path.write_bytes(jpeg_bytes()[:-32])
    Image.new("RGB", (64, 48), (128, 128, 128)).save(good_path)
    args = SimpleNamespace(resolution=1, data_device="cpu")
    bad, good = [camera_utils.loadCam(args, index, camera_info(path), 1.0)
                 for index, path in enumerate([bad_path, good_path])]
    test_cameras = [bad] if all_test_views_bad else [bad, good]
    scene = SimpleNamespace(gaussians=torch.nn.Module(), getTestCameras=lambda: test_cameras,
                            getTrainCameras=lambda: [good])
    infos, notices, rendered = [], [], []
    logger = SimpleNamespace(info=infos.append,
                             warning=lambda message, *args: notices.append(message % args))

    def render_valid(camera, *args, **kwargs):
        assert camera is good
        rendered.append(camera)
        return {"render": torch.zeros(3, 48, 64)}

    namespace = dict(torch=torch, Scene=object, psnr=psnr, prefilter_voxel=lambda *args: None)
    report = source_function("train.py", "training_report", namespace)
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image"):
        report(None, "scene", 1, torch.tensor(0.), torch.tensor(0.), l1_loss, 0., [1],
               scene, render_valid, (None, torch.zeros(3)), logger=logger)
    if all_test_views_bad:
        assert notices and "No decodable images" in notices[0]
        assert not any("Evaluating test" in message for message in infos)
        assert len(rendered) == 5
    else:
        summary = next(message for message in infos if "Evaluating test" in message)
        assert "1 valid views" in summary
        assert float(summary.split(" L1 ")[1].split(" PSNR ")[0]) == pytest.approx(128 / 255)
        assert len(rendered) == 6
    assert scene.gaussians.training


@pytest.mark.parametrize("filename", ["train.py", "render.py"])
def test_render_skips_corrupt_views_and_removes_stale_outputs(filename, camera_utils, monkeypatch, tmp_path):
    bad_path, good_path = tmp_path / "bad.jpg", tmp_path / "good.png"
    bad_path.write_bytes(jpeg_bytes()[:-32])
    Image.new("RGB", (64, 48), (128, 128, 128)).save(good_path)
    args = SimpleNamespace(resolution=1, data_device="cpu")
    paths = [bad_path] + [good_path] * 6
    cameras = [camera_utils.loadCam(args, index, camera_info(path), 1.0)
               for index, path in enumerate(paths)]
    output = tmp_path / "test/ours_1"
    for directory in ["renders", "gt", "errors"]:
        (output / directory).mkdir(parents=True)
        (output / directory / "00000.png").write_bytes(b"stale recovered output")
        (output / directory / "99999.png").write_bytes(b"orphaned output")
    rendered = []

    def render_valid(camera, *args, **kwargs):
        assert camera is not cameras[0]
        rendered.append(camera)
        return {"render": torch.zeros(3, 48, 64), "radii": torch.ones(2)}

    def save_image(tensor, path):
        pixels = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(path)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    namespace = dict(os=os, torch=torch, time=time, np=np, Path=Path, json=json,
                     makedirs=os.makedirs, tqdm=QuietProgress, render=render_valid,
                     prefilter_voxel=lambda *args: None,
                     torchvision=SimpleNamespace(utils=SimpleNamespace(save_image=save_image)))
    render_set = source_function(filename, "render_set", namespace)
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image"):
        render_set(str(tmp_path), "test", 1, cameras, None, None, torch.zeros(3))
    assert len(rendered) == 6
    expected = {f"{index:05d}.png" for index in range(1, 7)}
    assert {path.name for path in (output / "renders").iterdir()} == expected
    assert {path.name for path in (output / "gt").iterdir()} == expected
    if filename == "train.py":
        assert {path.name for path in (output / "errors").iterdir()} == expected
        assert json.loads((output / "per_view_count.json").read_text()) == {name: 2.0 for name in expected}


def test_synthetic_reader_retains_paths_and_preprocessing_only(monkeypatch, tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGBA", (24, 16), (90, 130, 180, 128)).save(path)
    (tmp_path / "transforms.json").write_text(json.dumps({
        "camera_angle_x": 0.8,
        "frames": [{"file_path": "frame.png", "transform_matrix": np.eye(4).tolist()}],
    }))
    # Execute the actual metadata-reader definitions without importing optional
    # CUDA/dataset extensions, which these functions do not use.
    source = ast.parse((ROOT / "scene/dataset_readers.py").read_text())
    nodes = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"CameraInfo", "readCamerasFromTransforms"}]
    namespace = dict(os=os, np=np, json=json, Path=Path, Image=Image, NamedTuple=NamedTuple,
                     tqdm=QuietProgress, Fore=SimpleNamespace(YELLOW=""), Style=SimpleNamespace(RESET_ALL=""),
                     focal2fov=focal2fov, fov2focal=fov2focal, open_image=open_image,
                     CorruptImageError=CorruptImageError, warn_corrupt_image=warn_corrupt_image)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<synthetic metadata>", "exec"), namespace)

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Synthetic metadata read decoded image pixels")

    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, "load", unexpected_decode)
        infos = namespace["readCamerasFromTransforms"](str(tmp_path), "transforms.json", True)
    assert len(infos) == 1 and infos[0].image is None
    assert infos[0].image_path == str(path)
    assert infos[0].image_options == {"background": [1, 1, 1]}
    assert (infos[0].width, infos[0].height) == (24, 16)


def to_tensor(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous().float() / 255


def metric_namespace(filename):
    source = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in {"readImages", "evaluate"}]

    class Metric(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, rendering, gt):
            return (rendering - gt).square().mean()

    namespace = dict(os=os, Path=Path, Image=Image, torch=torch, json=json,
                     tf=SimpleNamespace(to_tensor=to_tensor), tqdm=QuietProgress,
                     ssim=ssim, psnr=psnr, lpips_fn=Metric())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("filename", ["train.py", "metrics.py"])
def test_metrics_stream_pairs_and_preserve_results(filename, monkeypatch, tmp_path):
    renders = tmp_path / "test/ours_1/renders"
    gts = tmp_path / "test/ours_1/gt"
    renders.mkdir(parents=True)
    gts.mkdir()
    expected = []
    for index in range(3):
        name = "{}.png".format(index)
        Image.new("RGB", (16, 12), (30 + index * 20, 70, 110)).save(renders / name)
        Image.new("RGB", (16, 12), (60, 80 + index * 20, 120)).save(gts / name)
        with Image.open(renders / name) as rendering, Image.open(gts / name) as gt:
            a, b = to_tensor(rendering)[None], to_tensor(gt)[None]
            expected.append([ssim(a, b).item(), psnr(a, b).mean().item(), (a - b).square().mean().item()])
    namespace = metric_namespace(filename)
    original_open = Image.open
    opened = []

    def track_open(*args, **kwargs):
        opened.append(args[0])
        return original_open(*args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Image, "open", track_open)
        pairs = namespace["readImages"](renders, gts)
        assert not opened
        _, rendering, gt = next(pairs)
        assert len(opened) == 2
        assert rendering.device.type == gt.device.type == "cpu"
        pairs.close()
    if filename == "train.py":
        namespace["evaluate"](str(tmp_path), visible_count=[1, 1, 1], logger=SimpleNamespace(info=lambda *args: None))
    else:
        namespace["evaluate"]([str(tmp_path)])
    result = json.loads((tmp_path / "results.json").read_text())["ours_1"]
    np.testing.assert_allclose([result[key] for key in ["SSIM", "PSNR", "LPIPS"]],
                               np.mean(expected, axis=0), rtol=1e-5)


@pytest.mark.parametrize("filename", ["train.py", "metrics.py"])
def test_metrics_do_not_report_values_when_no_valid_views_remain(filename, tmp_path):
    (tmp_path / "test/ours_1/renders").mkdir(parents=True)
    (tmp_path / "test/ours_1/gt").mkdir()
    namespace = metric_namespace(filename)
    if filename == "train.py":
        namespace["evaluate"](str(tmp_path), visible_count=[],
                              logger=SimpleNamespace(warning=lambda *args: None))
    else:
        namespace["evaluate"]([str(tmp_path)])
    assert json.loads((tmp_path / "results.json").read_text()) == {"ours_1": {}}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_only_selected_image_moves_to_cuda(camera_utils, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (24, 16), (90, 130, 180)).save(path)
    args = SimpleNamespace(resolution=1, data_device="cuda")
    cameras = [camera_utils.loadCam(args, index, camera_info(path), 1.0) for index in range(100)]
    selected = cameras[3].load_image("cuda")
    assert selected.device.type == "cuda"
    assert all(camera._image is None for camera in cameras)
