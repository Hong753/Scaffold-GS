"""Camera parity, checkpoint selection and browser render scheduling on CPU."""

import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import socket
import struct
import threading
import time
from types import ModuleType, SimpleNamespace
import urllib.request

import numpy as np
from PIL import Image
import pytest
import torch

from utils.viewer_app import ClientState, ScaffoldViewer
from utils.viewer_renderer import make_view_camera
from utils.viewer_utils import (matrix_quaternion, overview_preset, quaternion_matrix,
                                read_camera_presets, read_model_config, render_dimensions,
                                resolve_checkpoint, world_to_camera)


ROOT = Path(__file__).resolve().parents[1]


def checkpoint(root, iteration=40000, native_name="point_cloud_scaffold.ply"):
    directory = root / "point_cloud" / f"iteration_{iteration}"
    directory.mkdir(parents=True)
    (root / "cfg_args").write_text("Namespace(feat_dim=32, n_offsets=10, appearance_dim=0)")
    (directory / native_name).write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float f_offset_0\n"
        b"property float f_anchor_feat_0\nend_header\n")
    for name in ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]:
        (directory / name).write_bytes(b"MLP fixture")
    return directory


def test_checkpoint_uses_latest_native_sidecar(tmp_path):
    checkpoint(tmp_path, 7000)
    latest = checkpoint(tmp_path, 40000)
    (latest / "point_cloud.ply").write_bytes(b"baked PLY must not be selected")
    root, directory, ply, config, iteration = resolve_checkpoint(tmp_path)
    assert root == tmp_path and directory == latest and iteration == 40000
    assert ply.name == "point_cloud_scaffold.ply" and config["n_offsets"] == 10
    assert resolve_checkpoint(latest)[4] == 40000
    assert resolve_checkpoint(tmp_path, 7000)[4] == 7000


def test_legacy_native_ply_supported(tmp_path):
    checkpoint(tmp_path, native_name="point_cloud.ply")
    assert resolve_checkpoint(tmp_path)[2].name == "point_cloud.ply"


def test_baked_only_ply_rejected(tmp_path):
    directory = checkpoint(tmp_path, native_name="point_cloud.ply")
    (directory / "point_cloud.ply").write_bytes(b"ply\nelement vertex 0\nproperty float f_dc_0\nend_header\n")
    with pytest.raises(ValueError, match="baked PLY"):
        resolve_checkpoint(tmp_path)


def test_missing_mlps_identified(tmp_path):
    directory = checkpoint(tmp_path)
    (directory / "cov_mlp.pt").unlink()
    with pytest.raises(FileNotFoundError, match="cov_mlp.pt"):
        resolve_checkpoint(tmp_path)


@pytest.mark.parametrize("config", ["__import__('os').system('bad')", "Namespace(x=print('bad'))", "Namespace(**{})"])
def test_config_is_not_evaluated(tmp_path, config):
    (tmp_path / "cfg_args").write_text(config)
    with pytest.raises(ValueError):
        read_model_config(tmp_path)


@pytest.mark.parametrize("q", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 2, 3, 4]])
def test_camera_rotations_roundtrip(q):
    rotation = quaternion_matrix(q)
    np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(rotation)), rotation, atol=1e-7)
    position = np.array([1.2, -3.4, 5.6])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = rotation, position
    np.testing.assert_allclose(world_to_camera(q, position) @ c2w, np.eye(4), atol=1e-6)


@pytest.mark.parametrize("width,height", [(1200, 800), (800, 1200)])
def test_viewer_projection_matches_native_camera(monkeypatch, width, height):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    scene = ModuleType("scene")
    scene.__path__ = [str(ROOT / "scene")]
    monkeypatch.setitem(sys.modules, "scene", scene)
    spec = importlib.util.spec_from_file_location("scene.cameras", ROOT / "scene/cameras.py")
    cameras = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scene.cameras", cameras)
    spec.loader.exec_module(cameras)
    q, position, fov = [1., 2., 3., 4.], [1.2, -3.4, 5.6], 0.8
    view = make_view_camera(q, position, fov, width, height, 3, "cpu", far=100.)
    rotation = quaternion_matrix(q)
    native = cameras.Camera(0, rotation, -rotation.T @ position, view.FoVx, fov,
                             None, None, "preset", 3, data_device="cpu", image_path="unused.jpg",
                             image_resolution=(width, height))
    torch.testing.assert_close(view.world_view_transform, native.world_view_transform, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view.full_proj_transform, native.full_proj_transform, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view.camera_center, native.camera_center, atol=1e-6, rtol=1e-6)
    assert view.uid == native.uid
    # A point on the optical axis projects to the exact image center.
    point = torch.tensor([*(np.asarray(position) + rotation[:, 2] * 4), 1.], dtype=torch.float32)
    projected = point @ view.full_proj_transform
    torch.testing.assert_close(projected[:2] / projected[3], torch.zeros(2), atol=1e-6, rtol=0)


def test_camera_presets_use_saved_training_appearance_ids(tmp_path):
    records = [dict(img_name="frame9", position=[1, 2, 3], rotation=np.eye(3).tolist(), height=800, fy=700),
               dict(img_name="frame2", position=[4, 5, 6], rotation=np.eye(3).tolist(), height=800, fy=700)]
    (tmp_path / "cameras.json").write_text(json.dumps(records))
    (tmp_path / "frame_selection.json").write_text(json.dumps({"selected_images": ["/data/frame2.jpg", "/data/frame9.jpg"]}))
    presets = read_camera_presets(tmp_path)
    assert [preset.appearance_id for preset in presets] == [1, 0]
    assert presets[0].fov == pytest.approx(2 * math.atan(800 / 1400))
    np.testing.assert_allclose(presets[0].position, [1, 2, 3])


def test_overview_looks_at_scene_center():
    center = np.array([2., 4., -1.])
    preset = overview_preset(center, 3., [0., 0., 1.])
    direction = (center - preset.position) / np.linalg.norm(center - preset.position)
    np.testing.assert_allclose(quaternion_matrix(preset.wxyz)[:, 2], direction, atol=1e-7)


@pytest.mark.parametrize("aspect,expected", [(1.5, (900, 600)), (0.5, (450, 900)), (1., (900, 900))])
def test_resolution_follows_browser_aspect(aspect, expected):
    assert render_dimensions(aspect, 900) == expected


class FakeRenderer:
    def __init__(self):
        self.iteration = 40000
        self.calls = []
        self.callback = None

    def render(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.callback:
            self.callback()
        return np.full((args[4], args[3], 3), 123, dtype=np.uint8), 10


def scheduler():
    app = ScaffoldViewer.__new__(ScaffoldViewer)
    app.renderer, app.fps, app.wake = FakeRenderer(), 30, threading.Event()
    images, downloads = [], []
    client = SimpleNamespace(scene=SimpleNamespace(set_background_image=lambda *args, **kwargs: images.append((args, kwargs))),
                             send_file_download=lambda *args: downloads.append(args))
    state = ClientState(client)
    state.controls = {key: SimpleNamespace(value=value) for key, value in dict(
        pause=False, resolution=640, moving=320, appearance=0, background=(0, 0, 0),
        scale=1., quality=85, status="").items()}
    state.pose = (np.array([1., 0., 0., 0.]), np.zeros(3), 0.8, 1.5)
    return app, state, images, downloads


def test_scheduler_refines_to_lossless_stationary_frame():
    app, state, images, _ = scheduler()
    state.moved_at = 10.
    assert app.render_pending(state, now=10.05)
    assert images[-1][1]["format"] == "jpeg"
    assert app.renderer.calls[-1][0][3] == 320
    assert app.render_pending(state, now=10.3)
    assert images[-1][1]["format"] == "png"
    assert app.renderer.calls[-1][0][3] == 640
    assert not app.render_pending(state, now=11.)
    assert len(app.renderer.calls) == 2


def test_motion_during_inference_does_not_starve_the_frame_stream():
    app, state, images, _ = scheduler()
    app.renderer.callback = lambda: app._invalidate(state)
    assert app.render_pending(state, now=10.)
    assert len(images) == 1
    app.renderer.callback = None
    assert app.render_pending(state, now=11.)
    assert len(images) == 2


def test_screenshot_is_full_resolution_png_even_when_paused():
    app, state, images, downloads = scheduler()
    state.controls["pause"].value = True
    assert not app.render_pending(state, now=10.)
    state.download = True
    assert app.render_pending(state, now=10.)
    assert images[-1][1]["format"] == "png"
    with Image.open(io.BytesIO(downloads[0][1])) as image:
        assert image.format == "PNG" and image.width == 640
    assert not state.download


def test_gpu_render_error_can_recover_after_settings_change():
    app, state, images, _ = scheduler()

    def fail():
        raise RuntimeError("CUDA out of memory")

    app.renderer.callback = fail
    assert not app.render_pending(state, now=10.)
    assert "CUDA out of memory" in state.controls["status"].value
    app.renderer.callback = None
    state.controls["resolution"].value = 320
    app._invalidate(state)
    assert app.render_pending(state, now=11.)
    assert images[-1][0][0].shape[1] == 320


def test_live_server_camera_and_gui_protocol(tmp_path):
    """Exercise actual Viser HTTP/WebSocket traffic with CPU test images."""
    import msgspec
    import websockets.sync.client
    import zstandard

    backend = FakeRenderer()
    backend.root, backend.anchor_count, backend.appearance_count = tmp_path, 10, 2
    backend.center, backend.radius, backend.up = np.zeros(3), 1., np.array([0., 0., 1.])
    backend.default_background = (0, 0, 0)
    backend.presets = [overview_preset(np.array([index, 0., 0.]), 1., backend.up) for index in range(2)]
    backend.presets[0].name, backend.presets[1].name = "frame1", "frame2"
    backend.presets[1].appearance_id = 1
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    app = ScaffoldViewer(backend, host="127.0.0.1", port=port, max_resolution=640, moving_resolution=320)
    worker = threading.Thread(target=app.run, daemon=True)
    worker.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}", timeout=3) as response:
            assert response.status == 200 and b"<html" in response.read().lower()
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", proxy=None,
                                           subprotocols=["viser-v1.1.1"], max_size=20_000_000) as connection:
            camera_message = dict(type="ViewerCameraMessage", wxyz=[1., 0., 0., 0.],
                                  position=[0., 0., -5.], fov=.8, near=.01, far=1000.,
                                  image_height=800, image_width=1200,
                                  look_at=[0., 0., 0.], up_direction=[0., -1., 0.])
            connection.send(msgspec.msgpack.encode(camera_message))
            widgets, formats = {}, []

            def receive_until(condition):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    raw = connection.recv(timeout=5)
                    _, compressed = struct.unpack("<QQ", raw[:16])
                    decoded = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16:16+compressed]))
                    for message in decoded["messages"]:
                        label = message.get("props", {}).get("label")
                        if label:
                            widgets[label] = message
                        if message["type"] == "BackgroundImageMessage" and message.get("rgb_data") is not None:
                            formats.append(message["format"])
                    if condition():
                        return
                raise AssertionError("Timed out waiting for a native viewer response")

            receive_until(lambda: "png" in formats and "Next view" in widgets)
            assert "Stationary resolution" in widgets and "Download PNG" in widgets
            assert backend.calls and backend.calls[-1][0][3] == 640
            connection.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage",
                uuid=widgets["Next view"]["uuid"], updates={"value": True})))
            receive_until(lambda: any(options["appearance_id"] == 1 for _, options in backend.calls))
            camera_message.update(image_width=600, image_height=900, position=[2., 3., 4.])
            connection.send(msgspec.msgpack.encode(camera_message))
            receive_until(lambda: any(args[3] < args[4] and np.allclose(args[1], [2, 3, 4])
                                      for args, _ in backend.calls))
    finally:
        app.stop()
        worker.join(timeout=3)
