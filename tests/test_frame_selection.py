"""Exercise temporal sharpness selection without CUDA or training extensions."""

import ast
from collections import namedtuple
from argparse import ArgumentParser
import io
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageFile
import pytest

from arguments import ModelParams
from utils import frame_selection
from utils.frame_selection import select_training_frames


ROOT = Path(__file__).resolve().parents[1]


def camera(path):
    return SimpleNamespace(image_path=str(path), image_name=path.stem)


def make_frames(directory, radii):
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    cameras = []
    for index, radius in enumerate(radii, 1):
        path = directory / f"frame{index}.png"
        source.filter(ImageFilter.GaussianBlur(radius)).save(path)
        cameras.append(camera(path))
    return cameras


def select(cameras, tmp_path, **kwargs):
    options = dict(stride=3, max_size=0, workers=2,
                   cache_dir=tmp_path / "cache", manifest_path=tmp_path / "model/frames.json")
    options.update(kwargs)
    return select_training_frames(cameras, **options)


def test_sharpest_per_group_natural_order_and_partial_tail(tmp_path):
    cameras = make_frames(tmp_path / "images", [3, 0, 1, 0, 1, 3, 3, 1, 0, 3])
    selected = select(list(reversed(cameras)), tmp_path)
    assert [item.image_name for item in selected] == ["frame2", "frame4", "frame9", "frame10"]
    payload = json.loads((tmp_path / "model/frames.json").read_text())
    assert payload["metric"] == "variance_of_laplacian"
    assert len(payload["frames"]) == 10
    assert payload["selected_images"] == [item.image_path for item in selected]
    for record in payload["frames"]:
        with Image.open(record["path"]) as source:
            gray = np.asarray(source.convert("L"))
        expected = cv2.Laplacian(gray, cv2.CV_64F).var()
        assert record["score"] == pytest.approx(expected)
    assert all(set(vars(item)) == {"image_path", "image_name"} for item in cameras)


def test_ties_keep_first_temporal_frame(tmp_path):
    cameras = []
    for index in [10, 11, 9]:
        path = tmp_path / f"frame{index}.png"
        Image.new("RGB", (16, 16), (100, 100, 100)).save(path)
        cameras.append(camera(path))
    assert select(cameras, tmp_path)[0].image_name == "frame9"


def test_cached_selection_does_not_decode_again(tmp_path, monkeypatch):
    cameras = make_frames(tmp_path / "images", [3, 0, 1, 1])
    expected = select(cameras, tmp_path)

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Cached selection decoded an image")

    monkeypatch.setattr(Image.Image, "load", unexpected_decode)
    assert select(cameras, tmp_path) == expected


def test_changing_source_invalidates_scores(tmp_path):
    cameras = make_frames(tmp_path / "images", [1, 3, 3])
    assert select(cameras, tmp_path)[0] is cameras[0]
    pixels = ((np.indices((64, 96)).sum(axis=0) % 2) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(cameras[2].image_path)
    assert select(cameras, tmp_path)[0] is cameras[2]


def test_reload_uses_exact_saved_subset_without_rescoring(tmp_path, monkeypatch):
    cameras = make_frames(tmp_path / "images", [3, 0, 1, 1])
    expected = select(cameras, tmp_path)

    def unexpected_score(*args, **kwargs):
        raise AssertionError("Reload rescored images and could change appearance IDs")

    monkeypatch.setattr(frame_selection, "_score_frame", unexpected_score)
    assert select(list(reversed(cameras)), tmp_path, stride=1, reuse_manifest=True) == expected


def test_corrupt_frames_are_excluded_without_regrouping(tmp_path, monkeypatch):
    cameras = make_frames(tmp_path / "images", [0] * 7)
    stream = io.BytesIO()
    Image.new("RGB", (96, 64), (100, 120, 140)).save(stream, format="JPEG")
    for index in [0, 3, 4, 5]:
        Path(cameras[index].image_path).write_bytes(stream.getvalue()[:-32])
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image") as messages:
        selected = select(cameras, tmp_path)
    assert len(messages) == 4
    assert selected == [cameras[1], cameras[6]]
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True


def test_all_corrupt_frames_stop_with_clear_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"broken image data")
    with pytest.warns(RuntimeWarning, match="Skipping corrupt image"), \
            pytest.raises(RuntimeError, match="No decodable training images"):
        select([camera(path)], tmp_path)


def test_permission_errors_are_not_silently_skipped(tmp_path, monkeypatch):
    cameras = make_frames(tmp_path / "images", [0])

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Image, "open", denied)
    with pytest.raises(PermissionError, match="frame1.png"):
        select(cameras, tmp_path)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_stride_one_preserves_input_without_touching_files(tmp_path):
    cameras = [camera(tmp_path / "missing2.png"), camera(tmp_path / "missing1.png")]
    assert select(cameras, tmp_path, stride=1) == cameras
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("options", [{"stride": 0}, {"workers": 0}, {"max_size": -1}])
def test_invalid_settings_rejected(tmp_path, options):
    with pytest.raises(ValueError, match="must be"):
        select([], tmp_path, **options)


def test_cli_parameters_are_saved_model_parameters():
    parser = ArgumentParser()
    params = ModelParams(parser)
    args = params.extract(parser.parse_args(["--frame_selection_stride", "3",
                                             "--frame_selection_max_size", "1600",
                                             "--frame_selection_workers", "2"]))
    assert (args.frame_selection_stride, args.frame_selection_max_size,
            args.frame_selection_workers) == (3, 1600, 2)


def test_scene_subsamples_train_only_and_preserves_reload_ids(tmp_path, monkeypatch):
    # Execute the real Scene constructor with only Gaussian/CUDA I/O stubbed.
    # Real selection, image decoding, cache and model manifest all run.
    monkeypatch.chdir(tmp_path)
    train = make_frames(tmp_path / "images", [3, 0, 1] * 3)
    test = make_frames(tmp_path / "heldout", [0, 0])
    (tmp_path / "sparse").mkdir()
    input_ply = tmp_path / "points.ply"
    input_ply.write_bytes(b"point cloud fixture")
    model = tmp_path / "model"
    model.mkdir()
    Info = namedtuple("Info", "train_cameras test_cameras nerf_normalization ply_path point_cloud")
    info = Info(train, test, {"radius": 1.0}, str(input_ply), None)

    class Gaussians:
        def set_appearance(self, count):
            self.appearance_count = count

        def create_from_pcd(self, *args):
            pass

        def load_ply_sparse_gaussian(self, *args):
            pass

        def load_mlp_checkpoints(self, *args):
            pass

    namespace = dict(os=os, random=random, json=json, GaussianModel=object, ModelParams=object,
                     sceneLoadTypeCallbacks={"Colmap": lambda *args, **kwargs: info},
                     cameraList_from_camInfos=lambda cameras, *args: cameras,
                     camera_to_JSON=lambda index, item: {"path": item.image_path},
                     select_training_frames=select_training_frames)
    source = ast.parse((ROOT / "scene/__init__.py").read_text())
    node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Scene")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "scene/__init__.py", "exec"), namespace)
    args = SimpleNamespace(source_path=str(tmp_path), model_path=str(model), images="images",
                           eval=True, lod=0, frame_selection_stride=3,
                           frame_selection_max_size=0, frame_selection_workers=2)
    gaussians = Gaussians()
    scene = namespace["Scene"](args, gaussians, shuffle=False)
    assert [item.image_name for item in scene.getTrainCameras()] == ["frame2", "frame5", "frame8"]
    assert scene.getTestCameras() == test
    assert len(train) == 9 and gaussians.appearance_count == 3
    assert len(json.loads((model / "cameras.json").read_text())) == 5
    manifest = json.loads((model / "frame_selection.json").read_text())
    assert {frame["path"] for frame in manifest["frames"]} == {item.image_path for item in train}

    def unexpected_score(*args, **kwargs):
        raise AssertionError("Model reload must use saved training frames")

    monkeypatch.setattr(frame_selection, "_score_frame", unexpected_score)
    reloaded = namespace["Scene"](args, Gaussians(), load_iteration=30000, shuffle=False)
    assert reloaded.getTrainCameras() == scene.getTrainCameras()
    assert reloaded.getTestCameras() == test
