"""Real COLMAP fixtures and a live HTTP/WebSocket test for the CPU pose audit."""

from argparse import Namespace
import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import socket
import re
import struct
import threading
import time
import urllib.request
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from utils.pose_audit_data import PoseAuditData, image_resolution, read_observations
from utils.viewer_renderer import make_view_camera


def make_dataset(tmp_path, suffix="bin", distortion=.06, bad_pose=False, yaw=0.):
    source = tmp_path / "dataset"
    sparse, images, model = source / "sparse/0", source / "images", tmp_path / "run"
    sparse.mkdir(parents=True)
    images.mkdir()
    model.mkdir()
    names = ["frame1.png", "frame2.png", "frame10.png", "frame11.png"]
    ids = np.array([55, 11, 2**54 + 1], dtype=np.int64)
    xyz = np.array([[.4, .2, 4.], [-.8, -.5, 5.], [.9, .4, 6.]])
    colors = [(220, 130, 70), (50, 200, 110), (80, 160, 240)]
    matrix = np.array([[250., 0, 160.], [0, 250., 120.], [0, 0, 1.]])
    d = np.array([distortion, 0., 0., 0.])
    qvec = [np.cos(yaw / 2), 0., np.sin(yaw / 2), 0.]
    rotation = cv2.Rodrigues(np.array([0., yaw, 0.]))[0]
    poses, observations = [], []
    for i, name in enumerate(names):
        translation = -rotation @ np.array([i * .1, 0., 0.])
        obs = cv2.projectPoints(xyz @ rotation.T + translation, np.zeros(3), np.zeros(3), matrix, d)[0].reshape(-1, 2)
        poses.append(translation + ([.1, 0, 0] if bad_pose and i == 1 else np.zeros(3)))
        observations.append(obs)
        image = Image.new("RGB", (320, 240), (29, 42, 58))
        draw = ImageDraw.Draw(image)
        for x in range(0, 320, 20):
            draw.line((x, 0, x, 239), fill=(45, 60, 76))
        for j, (u, v) in enumerate(obs):
            draw.ellipse((u - 4.5, v - 4.5, u + 3.5, v + 3.5), fill=colors[j])
        image.save(images / name)
    if suffix == "bin":
        with (sparse / "cameras.bin").open("wb") as stream:
            stream.write(struct.pack("<QiiQQdddd", 1, 1, 2, 320, 240, 250., 160., 120., distortion))
        with (sparse / "images.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", len(names)))
            for i, name in enumerate(names):
                stream.write(struct.pack("<idddddddi", i + 1, *qvec, *poses[i], 1))
                stream.write(name.encode() + b"\0")
                stream.write(struct.pack("<Q", len(ids) + 1))
                for xy, point_id in zip(observations[i], ids):
                    stream.write(struct.pack("<ddq", *xy, int(point_id)))
                stream.write(struct.pack("<ddq", 10., 10., -1))
        with (sparse / "points3D.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", len(ids)))
            for j, point_id in enumerate(ids):
                stream.write(struct.pack("<QdddBBBd", int(point_id), *xyz[j], *colors[j], .01))
                stream.write(struct.pack("<Q", 4))
                for i in range(4):
                    stream.write(struct.pack("<ii", i + 1, j))
    else:
        (sparse / "cameras.txt").write_text(f"# Camera\n1 SIMPLE_RADIAL 320 240 250 160 120 {distortion}\n")
        lines = ["# Images"]
        for i, name in enumerate(names):
            lines.append(f"{i + 1} {' '.join(map(str, qvec))} {' '.join(map(str, poses[i]))} 1 {name}")
            lines.append(" ".join(f"{u} {v} {int(point_id)}" for (u, v), point_id in zip(observations[i], ids)) + " 10 10 -1")
        (sparse / "images.txt").write_text("\n".join(lines) + "\n")
        (sparse / "points3D.txt").write_text("\n".join(
            f"{int(point_id)} {' '.join(map(str, xyz[j]))} {' '.join(map(str, colors[j]))} 0.01 1 {j} 2 {j}"
            for j, point_id in enumerate(ids)) + "\n")
    config = Namespace(source_path=str(source), images="images", resolution=2, eval=True, lod=0,
                       frame_selection_stride=3, undistort_images=True,
                       undistort_cache_dir=str(tmp_path / "cache"))
    (model / "cfg_args").write_text(str(config))
    (model / "frame_selection.json").write_text(json.dumps({"selected_images": [str(images / names[i]) for i in (1, 2)]}))
    return source, model


@pytest.mark.parametrize("suffix", ["bin", "txt"])
def test_indexing_is_metadata_only_and_observations_are_loaded_on_selection(tmp_path, monkeypatch, suffix):
    source, model = make_dataset(tmp_path, suffix)
    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, "load", lambda *a, **k: pytest.fail("Setup decoded image pixels"))
        data = PoseAuditData(model)
    assert [frame.name for frame in data.frames] == ["frame1.png", "frame2.png", "frame10.png", "frame11.png"]
    assert [frame.split for frame in data.frames] == ["test", "train", "train", "unused"]
    assert all(not hasattr(frame, "xys") and not hasattr(frame, "image") for frame in data.frames)
    assert not (tmp_path / "cache").exists()
    xys, ids = read_observations(data.images_file, suffix, data.frames[1])
    assert xys.shape == (4, 2) and ids.tolist() == [55, 11, 2**54 + 1, -1]
    np.testing.assert_allclose(data.frames[1].center, [.1, 0, 0])
    overlay, pixels, xyz, details = data.inspect(1)
    assert overlay.shape == pixels.shape == (120, 160, 3)
    assert xyz.shape == (3, 3) and details["associated_points"] == 3
    assert details["colmap_reprojection_px"]["max"] < 1e-8
    assert details["training_projection_px"]["max"] < .1
    assert details["behind_camera"] == 0


def test_camera_pose_perturbation_is_visible_in_reprojection_statistics(tmp_path):
    source, model = make_dataset(tmp_path, bad_pose=True)
    data = PoseAuditData(model)
    _, _, _, details = data.inspect(1)
    assert details["colmap_reprojection_px"]["median"] > 4
    assert details["training_projection_px"]["median"] > 2


@pytest.mark.parametrize("yaw", [0., .25])
def test_native_projection_uses_actual_cuda_pixel_center_convention(tmp_path, yaw):
    source, model = make_dataset(tmp_path, distortion=0., yaw=yaw)
    data = PoseAuditData(model)
    frame = data.frames[1]
    fov = 2 * np.arctan(240 / 500)
    view = make_view_camera(frame.wxyz, frame.center, fov, 160, 120, 0, "cpu")
    points = torch.from_numpy(np.column_stack((data.xyz, np.ones(len(data.xyz))))).float()
    clip = points @ view.full_proj_transform
    ndc = (clip[:, :2] / clip[:, 3:4]).numpy()
    cuda_pixels = ((ndc + 1) * np.array([160, 120]) - 1) / 2
    camera = data.xyz @ frame.rotation.T + frame.tvec
    expected = camera[:, :2] / camera[:, 2:3] * 125 + [79.5, 59.5]
    np.testing.assert_allclose(cuda_pixels, expected, atol=1e-5)
    _, _, _, details = data.inspect(1)
    assert details["training_projection_px"]["max"] < 1e-8


def test_images_without_matching_3d_points_still_display(tmp_path):
    source, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    data.ids, data.xyz, data.rgb = np.empty(0, np.int64), np.empty((0, 3)), np.empty((0, 3), np.uint8)
    overlay, pixels, xyz, details = data.inspect(1)
    assert overlay.shape == pixels.shape and xyz.shape == (0, 3)
    assert details["associated_points"] == 0 and details["training_projection_px"]["median"] is None


@pytest.mark.parametrize("observations,reason", [
    ("", "no stored 2D feature observations"),
    ("10 10 -1", "none are associated with a triangulated"),
    ("10 10 999", "IDs absent from the loaded")])
def test_missing_correspondences_are_explained_and_next_match_does_not_decode_images(tmp_path, monkeypatch, observations, reason):
    source, model = make_dataset(tmp_path, suffix="txt")
    path = source / "sparse/0/images.txt"
    lines = path.read_text().splitlines()
    lines[4] = observations
    path.write_text("\n".join(lines) + "\n")
    data = PoseAuditData(model)
    _, _, _, details = data.inspect(1)
    assert reason in details["correspondence_status"]
    assert details["associated_points"] == 0
    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, "load", lambda *a, **k: pytest.fail("Search decoded RGB pixels"))
        assert data.next_with_matches(1, [1, 2]) == 2
        assert data.next_with_matches(1, [1]) is None
        data.association_counts[2] = 0
        assert data.next_with_matches(1, [1, 2]) is None


@pytest.mark.parametrize("space", ["Original image", "Training input"])
def test_cloud_projection_works_without_correspondences_and_does_not_invent_errors(tmp_path, space):
    source, model = make_dataset(tmp_path, distortion=0.)
    data = PoseAuditData(model)
    data.frames[1].count = 0
    overlay, pixels, _, details = data.inspect(1, space, mode="Cloud projection")
    assert details["associated_points"] == 0 and details["cloud_overlay_points"] == 3
    assert details["training_projection_px"]["median"] is None
    assert details["colmap_reprojection_px"]["median"] is None
    assert not np.array_equal(overlay, pixels)
    for point in data.xyz:
        camera = point @ data.frames[1].rotation.T + data.frames[1].tvec
        x, y = np.round(camera[:2] / camera[2] * 125 + [79.5, 59.5]).astype(int)
        patch = overlay[y - 3:y + 4, x - 3:x + 4]
        assert ((patch == [255, 177, 49]).all(axis=2)).any()


def test_camera_picking_is_screen_based_and_includes_unsampled_and_overlapping_cameras():
    from utils.pose_audit_app import pick_cameras
    positions = np.array([[0., 0, 5], [0., 0, -1], [.5, 0, 5], [0., 0, 3]])
    assert pick_cameras(positions, [1, 0, 0, 0], np.zeros(3), np.pi / 2, 1000, 1000, [.5, .5]) == [3, 0]
    assert pick_cameras(positions, [1, 0, 0, 0], np.zeros(3), np.pi / 2, 1000, 1000, [.55, .5]) == [2]
    assert pick_cameras(positions, [1, 0, 0, 0], np.zeros(3), np.pi / 2, 1000, 1000, [.9, .1]) == []
    yaw = .4
    rotation = cv2.Rodrigues(np.array([0., yaw, 0.]))[0]
    from utils.viewer_utils import matrix_quaternion
    translated = np.array([1., 2., 3.])
    world = positions @ rotation.T + translated
    assert pick_cameras(world, matrix_quaternion(rotation), translated, np.pi / 2, 1000, 1000, [.55, .5]) == [2]


def test_image_inspector_keeps_full_resolution_and_native_zoom_scrolls():
    from utils.pose_audit_app import image_html
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 1200), "navy").save(buffer, format="PNG")
    page = image_html(buffer.getvalue(), '<frame "1">', 1600, "100%")
    encoded = re.search(r"base64,([^\"]+)", page).group(1)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (1600, 1200)
    assert "width:1600px" in page and "overflow:auto" in page and "max-width:none" in page
    assert '&lt;frame &quot;1&quot;&gt;' in page
    assert "width:3200px" in image_html(buffer.getvalue(), "frame", 1600, "200%")


def test_points_behind_camera_are_reported_and_excluded_from_overlay_statistics(tmp_path):
    source, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    data.xyz[0, 2] = -5
    _, _, xyz, details = data.inspect(1)
    assert details["behind_camera"] == 1 and len(xyz) == 2


def test_wrong_rgb_can_have_zero_colmap_error_and_report_explains_limitation(tmp_path):
    source, model = make_dataset(tmp_path)
    Image.new("RGB", (320, 240), "red").save(source / "images/frame2.png")
    data = PoseAuditData(model)
    _, pixels, _, details = data.inspect(1, "Original image")
    assert pixels[:, :, 0].min() == 255 and pixels[:, :, 1:].max() == 0
    assert details["colmap_reprojection_px"]["max"] < 1e-8
    report = data.export_report(tmp_path / "report", count=2)
    page = report.read_text()
    assert "does not prove" in page and "Original image" in page and "Training input" in page
    assert len(list(report.parent.glob("*.png"))) == 4


def test_report_records_missing_images_instead_of_silently_dropping_them(tmp_path):
    source, model = make_dataset(tmp_path)
    (source / "images/frame2.png").unlink()
    data = PoseAuditData(model)
    report = data.export_report(tmp_path / "report", count=2)
    records = json.loads((report.parent / "summary.json").read_text())
    assert "error" in records[0] and records[0]["image"] == "frame2.png"
    assert "frame2.png" in report.read_text()


def test_current_basename_mapping_and_dimension_mismatch_are_explicit(tmp_path):
    source, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    data.frames[1].name = "clip_a/frame2.png"
    data.collisions.add("frame2.png")
    Image.new("RGB", (320, 250), "gray").save(source / "images/frame2.png")
    _, _, _, details = data.inspect(1)
    assert details["training_path"] == str(source / "images/frame2.png")
    assert len(details["warnings"]) == 3


def test_dataset_only_audit_does_not_claim_saved_training_membership(tmp_path):
    source, _ = make_dataset(tmp_path)
    data = PoseAuditData(source_path=source)
    assert set(data.groups()) == {"All COLMAP images"}
    assert all(frame.split == "source" for frame in data.frames)
    assert not data.undistorted


@pytest.mark.parametrize("requested,expected", [(-1, (1600, 900)), (1, (3840, 2160)),
                                                  (2, (1920, 1080)), (1000, (1000, 562))])
def test_resolution_matches_training(requested, expected):
    assert image_resolution((3840, 2160), requested) == expected


def test_pose_audit_real_http_websocket_selection_and_png_download(tmp_path):
    import msgspec
    import websockets.sync.client
    import zstandard
    from utils.pose_audit_app import PoseAuditApp

    source, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    app = PoseAuditApp(data, host="127.0.0.1", port=port, max_points=100, max_frustums=1)
    worker = threading.Thread(target=app.run, daemon=True)
    worker.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}", timeout=3) as response:
            assert response.status == 200 and b"<html" in response.read().lower()
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", proxy=None,
                                           subprotocols=["viser-v1.1.1"], max_size=20_000_000) as connection:
            connection.send(msgspec.msgpack.encode(dict(type="ViewerCameraMessage", wxyz=[1., 0., 0., 0.],
                position=[0., 0., -5.], fov=.8, near=.01, far=1000., image_height=800, image_width=1200,
                look_at=[0., 0., 0.], up_direction=[0., -1., 0.])))
            widgets, values, types, downloads = {}, {}, set(), []

            def receive_until(condition):
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline:
                    raw = connection.recv(timeout=6)
                    _, compressed = struct.unpack("<QQ", raw[:16])
                    decoded = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16:16 + compressed]))
                    for message in decoded["messages"]:
                        types.add(message["type"])
                        label = message.get("props", {}).get("label")
                        if label:
                            widgets[label] = message
                        if message["type"] == "GuiUpdateMessage":
                            values.setdefault(message["uuid"], {}).update(message["updates"])
                        if message["type"] == "FileTransferStartDownload":
                            downloads.append(message)
                    if condition():
                        return
                raise AssertionError("Timed out waiting for pose audit UI")

            def selected(name):
                widget = widgets.get("Selected image")
                return widget and values.get(widget["uuid"], {}).get("value") == name

            receive_until(lambda: "Next image" in widgets and selected("frame2.png"))
            assert {"PointCloudMessage", "CameraFrustumMessage", "GuiHtmlMessage", "GuiPanelMessage", "ScenePointerEnableMessage"}.issubset(types)
            state = next(iter(app.clients.values()))
            assert state.cloud.point_shape == "circle" and state.cloud.point_size < data.radius * .0003
            # Zoom only changes presentation: it must not decode another image.
            rendered = state.rendered
            connection.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage", uuid=widgets["Image zoom"]["uuid"], updates={"value": "100%"})))
            receive_until(lambda: "width:160px" in state.controls["image"].content)
            assert state.rendered == rendered
            connection.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage", uuid=widgets["Next image"]["uuid"], updates={"value": True})))
            receive_until(lambda: selected("frame10.png"))
            connection.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage", uuid=widgets["Image space"]["uuid"], updates={"value": "Original image"})))
            connection.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage", uuid=widgets["Download overlay PNG"]["uuid"], updates={"value": True})))
            receive_until(lambda: len(downloads) > 0)
            assert downloads[-1]["filename"].endswith("_raw.png")
            # Click an unsampled camera outside the current training subset.
            # The only sampled frustum is frame1: frame11 must still be pickable.
            connection.send(msgspec.msgpack.encode(dict(type="ViewerCameraMessage", wxyz=[1., 0., 0., 0.],
                position=[.3, 0., -1.], fov=.8, near=.01, far=1000., image_height=800, image_width=1200,
                look_at=[.3, 0., 0.], up_direction=[0., -1., 0.])))
            connection.send(msgspec.msgpack.encode(dict(type="ScenePointerMessage", event_type="click",
                ray_origin=[.3, 0., -1.], ray_direction=[0., 0., 1.], screen_pos=[[.5, .5]], modifier=None)))
            receive_until(lambda: selected("frame11.png"))
            assert any("frame11.png" in label for label in state.candidates)
            # A missing image must clear the displayed pixels and PNG, including at zoom.
            (source / "images/frame11.png").unlink()
            connection.send(msgspec.msgpack.encode(dict(type="GuiUpdateMessage", uuid=widgets["Marks"]["uuid"], updates={"value": "None"})))
            receive_until(lambda: state.png == b"" and "Could not inspect" in state.controls["alignment"].content)
            assert "base64," not in state.controls["image"].content
            assert state.frustum is None and state.points is None
            app._display(state)  # Changing display options cannot restore stale highlights.
    finally:
        app.stop()
        worker.join(timeout=3)


@contextmanager
def audit_runtime(data):
    """Exercise the real event loop while connect/decode callbacks are paused."""
    from utils.pose_audit_app import PoseAuditApp

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    app = PoseAuditApp(data, host="127.0.0.1", port=port, max_points=100, max_frustums=4)
    errors = []

    def run():
        try:
            app.run()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield app, port
    finally:
        app.stop()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert not errors, f"Audit loop crashed: {errors}"


@contextmanager
def audit_connection(port):
    import msgspec
    import websockets.sync.client

    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", proxy=None,
            subprotocols=["viser-v1.1.1"], max_size=20_000_000) as connection:
        connection.send(msgspec.msgpack.encode(dict(type="ViewerCameraMessage", wxyz=[1., 0., 0., 0.],
            position=[0., 0., -5.], fov=.8, near=.01, far=1000., image_height=800, image_width=1200,
            look_at=[0., 0., 0.], up_direction=[0., -1., 0.])))
        yield connection


def wait_for(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for audit lifecycle condition")
        threading.Event().wait(.005)


def test_slow_connect_does_not_process_partial_controls(tmp_path, monkeypatch):
    _, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    entered, release = threading.Event(), threading.Event()
    with audit_runtime(data) as (app, port):
        original = app._add_scene

        def slow_scene(state):
            entered.set()
            assert release.wait(5)
            original(state)

        monkeypatch.setattr(app, "_add_scene", slow_scene)
        try:
            with audit_connection(port):
                assert entered.wait(3)
                assert not app.clients
                state = next(iter(app._initializing_clients.values()))
                assert state.controls == {} and not state.ready
                # This raised KeyError('index') in the reported version.
                assert app.process(state) is False
                release.set()
                wait_for(lambda: state.ready and bool(state.png))
                assert app.clients[state.client.client_id] is state and "index" in state.controls
        finally:
            release.set()


@pytest.mark.parametrize("phase", ["scene", "gui"])
def test_disconnect_during_setup_cannot_publish_stale_client_and_reconnect_works(tmp_path, monkeypatch, phase):
    _, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    entered, release = threading.Event(), threading.Event()
    with warnings.catch_warnings(record=True) as captured, audit_runtime(data) as (app, port):
        warnings.simplefilter("always")
        name = "_add_scene" if phase == "scene" else "_initialize"
        original = getattr(app, name)

        def paused(state):
            if phase == "gui":
                original(state)
            entered.set()
            assert release.wait(5)
            if phase == "scene":
                original(state)

        monkeypatch.setattr(app, name, paused)
        try:
            with audit_connection(port):
                assert entered.wait(3)
                old = next(iter(app._initializing_clients.values()))
                assert not old.ready and not app.clients
            wait_for(lambda: not app.server.get_clients())
            release.set()
            wait_for(lambda: not app._initializing_clients and not old.connected)
            assert not app.clients and not old.ready
            monkeypatch.setattr(app, name, original)
            with audit_connection(port):
                wait_for(lambda: bool(app.clients) and bool(next(iter(app.clients.values())).png))
                current = next(iter(app.clients.values()))
                assert current.client.client_id != old.client.client_id and current.ready
        finally:
            release.set()
        assert not any("closed connection" in str(item.message) for item in captured)


def test_disconnect_while_decoding_drops_result_and_server_accepts_new_client(tmp_path, monkeypatch):
    _, model = make_dataset(tmp_path)
    data = PoseAuditData(model)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = data.inspect

    def slow_inspect(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        result = original(*args, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(data, "inspect", slow_inspect)
    with warnings.catch_warnings(record=True) as captured, audit_runtime(data) as (app, port):
        warnings.simplefilter("always")
        try:
            with audit_connection(port):
                assert entered.wait(3)
                old = next(iter(app.clients.values()))
            wait_for(lambda: not old.connected)
            release.set()
            assert completed.wait(3)
            assert not old.ready and not old.png and not app.clients
            monkeypatch.setattr(data, "inspect", original)
            with audit_connection(port):
                wait_for(lambda: bool(app.clients) and bool(next(iter(app.clients.values())).png))
        finally:
            release.set()
        assert not any("closed connection" in str(item.message) for item in captured)
