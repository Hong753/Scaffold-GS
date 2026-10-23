"""CPU pose audit: 3D cameras/points with lazy, per-client image overlays."""

from dataclasses import dataclass, field
import io
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import viser

from utils.pose_audit_data import COLORS
from utils.viewer_utils import overview_preset


@dataclass
class AuditClient:
    client: object
    controls: dict = field(default_factory=dict)
    indices: list = field(default_factory=list)
    lock: object = field(default_factory=threading.RLock)
    revision: int = 0
    rendered: int = -1
    connected: bool = True
    download: bool = False
    png: bytes = b""
    filename: str = "pose_overlay.png"
    frustum: object = None
    points: object = None


class PoseAuditApp:
    def __init__(self, data, host="0.0.0.0", port=8081, max_points=100000, max_frustums=120):
        self.data, self.clients = data, {}
        self.wake, self.stopped = threading.Event(), threading.Event()
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(dark_mode=True, control_width="large")
        self.server.gui.main_panel.dock_right()
        self.server.scene.set_up_direction(data.up)
        chosen = np.linspace(0, len(data.xyz) - 1, min(len(data.xyz), max_points), dtype=int)
        self.server.scene.add_point_cloud("/sfm_points", data.xyz[chosen].astype(np.float32),
                                          data.rgb[chosen], point_size=data.radius * .0018, precision="float32")
        self.server.scene.add_point_cloud("/camera_centers", data.positions.astype(np.float32),
                                          np.array([COLORS[frame.split] for frame in data.frames], np.uint8),
                                          point_size=data.radius * .006, precision="float32")
        if len(data.frames) > 1:
            segments = np.stack((data.positions[:-1], data.positions[1:]), axis=1).astype(np.float32)
            self.server.scene.add_line_segments("/trajectory", segments, (85, 118, 148),
                                                thickness=1., thickness_units="screen")
        for index in np.linspace(0, len(data.frames) - 1, min(max_frustums, len(data.frames)), dtype=int):
            frame = data.frames[index]
            camera = data.cameras[frame.camera_id]
            k = data.options[frame.camera_id]["colmap_calibration"]["matrix"]
            frustum = self.server.scene.add_camera_frustum(
                f"/cameras/{frame.id}", fov=2 * np.arctan(camera.height / (2 * k[1][1])),
                aspect=camera.width * k[1][1] / (camera.height * k[0][0]), scale=data.radius * .012,
                color=COLORS[frame.split], position=frame.center, wxyz=frame.wxyz,
                thickness=1., thickness_units="screen")
            frustum.on_click(lambda event, index=int(index): self.select_frame(event.client_id, index))
        self.server.on_client_connect(self._connect)
        self.server.on_client_disconnect(self._disconnect)

    def _dirty(self, state):
        with state.lock:
            state.revision += 1
        self.wake.set()

    def _overview(self, state):
        preset = overview_preset(self.data.center, self.data.radius, self.data.up)
        with state.client.atomic():
            state.client.camera.position = preset.position
            state.client.camera.wxyz = preset.wxyz
            state.client.camera.fov = preset.fov
            state.client.camera.look_at = self.data.center

    def _look(self, state):
        with state.lock:
            index = state.indices[int(state.controls["index"].value)]
        frame = self.data.frames[index]
        camera = self.data.cameras[frame.camera_id]
        k = self.data.options[frame.camera_id]["colmap_calibration"]["matrix"]
        with state.client.atomic():
            state.client.camera.position = frame.center
            state.client.camera.wxyz = frame.wxyz
            state.client.camera.fov = 2 * np.arctan(camera.height / (2 * k[1][1]))
            state.client.camera.look_at = frame.center + frame.rotation[2] * self.data.radius * .2

    def _connect(self, client):
        state = AuditClient(client)
        groups = self.data.groups()
        default = "Training subset" if "Training subset" in groups else "All COLMAP images"
        state.indices = groups[default]
        self.clients[client.client_id] = state
        gui = client.gui
        gui.add_markdown(f"## Image / pose audit\n{len(self.data.frames):,} cameras · {len(self.data.xyz):,} SfM points\n\n"
                         "Green: training · purple: test · gray: other cameras. Click a camera frustum to inspect it.")
        if self.data.root is None:
            gui.add_markdown("Dataset-only audit: no saved training subset or preprocessing configuration was supplied.")
        if not self.data.selection_known:
            gui.add_markdown("**Saved selection manifest missing:** training membership below represents candidates.")
        controls = state.controls
        with gui.add_folder("Choose image"):
            controls["group"] = gui.add_dropdown("Views", list(groups), initial_value=default)
            controls["index"] = gui.add_slider("Image index", 0, max(1, len(state.indices) - 1), 1, 0,
                                               disabled=len(state.indices) == 1)
            previous, following = gui.add_button("Previous image"), gui.add_button("Next image")
            controls["find"] = gui.add_text("Filename", "", hint="COLMAP filename or stem, e.g. 00961")
            find = gui.add_button("Find image")
            controls["name"] = gui.add_text("Selected image", "", disabled=True)
            look, home = gui.add_button("Look from camera"), gui.add_button("Scene overview")
        with gui.add_folder("Inspect alignment"):
            controls["space"] = gui.add_dropdown("Image space", ["Training input", "Original image"],
                                                  initial_value="Training input" if self.data.root else "Original image")
            controls["mode"] = gui.add_dropdown("Marks", ["Both", "Observations", "Projections", "None"],
                                                 initial_value="Both")
            controls["limit"] = gui.add_slider("Overlay points", 50, 2000, 50, 500)
            controls["photo"] = gui.add_checkbox("Image on selected frustum", False)
            download = gui.add_button("Download overlay PNG")
            controls["stats"] = gui.add_markdown("Loading image…")
            controls["image"] = gui.add_image(np.zeros((16, 24, 3), np.uint8), label="Image overlay", format="png")
        gui.add_markdown("**Cyan circles:** COLMAP feature observations. **Orange crosses:** projected 3D points. "
                         "**Red lines:** residuals. Download the PNG to inspect at training resolution.\n\n"
                         "Low residuals show COLMAP self-consistency. Check that marks actually land on the correct "
                         "features in the RGB image; low residuals alone do not verify frame identities or global accuracy.")

        @controls["group"].on_update
        def group_changed(_):
            with state.lock:
                previous = state.indices[min(int(controls["index"].value), len(state.indices) - 1)]
                state.indices = groups[controls["group"].value]
                controls["index"].max = max(1, len(state.indices) - 1)
                controls["index"].disabled = len(state.indices) == 1
                controls["index"].value = state.indices.index(previous) if previous in state.indices else 0
            self._dirty(state)

        for key in ("index", "space", "mode", "limit", "photo"):
            controls[key].on_update(lambda _, state=state: self._dirty(state))

        @previous.on_click
        def prev(_):
            controls["index"].value = max(0, int(controls["index"].value) - 1)
            self._dirty(state)

        @following.on_click
        def next_image(_):
            controls["index"].value = min(len(state.indices) - 1, int(controls["index"].value) + 1)
            self._dirty(state)

        @find.on_click
        def find_image(_):
            query = controls["find"].value.strip()
            matches = [i for i, frame in enumerate(self.data.frames)
                       if query in (frame.name, Path(frame.name).name, Path(frame.name).stem)]
            if len(matches) == 1:
                self.select_frame(client.client_id, matches[0])
            else:
                controls["stats"].content = f"Found {len(matches)} matches. Enter a unique COLMAP filename."

        look.on_click(lambda _: self._look(state))
        home.on_click(lambda _: self._overview(state))

        @download.on_click
        def download_image(_):
            with state.lock:
                state.download = True
            self._dirty(state)

        self._overview(state)
        self._dirty(state)

    def select_frame(self, client_id, index):
        state = self.clients.get(client_id)
        if state is None:
            return
        controls = state.controls
        with state.lock:
            if index not in state.indices:
                controls["group"].value = "All COLMAP images"
                state.indices = self.data.groups()["All COLMAP images"]
                controls["index"].max = max(1, len(state.indices) - 1)
                controls["index"].disabled = len(state.indices) == 1
            controls["index"].value = state.indices.index(index)
        self._dirty(state)

    def _disconnect(self, client):
        state = self.clients.pop(client.client_id, None)
        if state:
            state.connected = False

    def process(self, state):
        with state.lock:
            if not state.connected or state.revision == state.rendered:
                return False
            revision = state.revision
            index = state.indices[min(int(state.controls["index"].value), len(state.indices) - 1)]
            space, mode = state.controls["space"].value, state.controls["mode"].value
            limit, photo = int(state.controls["limit"].value), state.controls["photo"].value
        frame = self.data.frames[index]
        try:
            overlay, pixels, associated, details = self.data.inspect(index, space, limit, mode)
            buffer = io.BytesIO()
            Image.fromarray(overlay).save(buffer, format="PNG", compress_level=1)
            thumbnail = Image.fromarray(overlay)
            thumbnail.thumbnail((1000, 1000))
            def metric(record):
                return "unavailable" if record["median"] is None else f"{record['median']:.3f} / {record['p95']:.3f} px"
            stats = (f"**{frame.split}** · RGB {details['source_size']} · training {details['training_size']}\n\n"
                     f"Associated points: **{details['associated_points']:,}** · valid training points: {details['valid_training_points']:,} · behind camera: {details['behind_camera']}\n\n"
                     f"COLMAP median / p95: **{metric(details['colmap_reprojection_px'])}** (original calibration size)\n\n"
                     f"Training median / p95: **{metric(details['training_projection_px'])}** (training pixels)\n\n"
                     f"Points inside native near cull (z ≤ 0.2 scene units): {details['within_native_near_cull']}")
            if "previous_translation" in details:
                stats += (f"\n\nPrevious-frame motion: {details['previous_translation']:.4g} SfM units · "
                          f"{details['previous_rotation_degrees']:.2f}°")
            if details["warnings"]:
                stats += "\n\n" + "\n\n".join(f"**Check:** {warning}" for warning in details["warnings"])
            with state.lock:
                if not state.connected or revision != state.revision:
                    return False
                state.controls["name"].value = frame.name
                state.controls["stats"].content = stats
                state.controls["image"].image = np.asarray(thumbnail)
                width, height = details["training_size"]
                k = self.data.options[frame.camera_id]["colmap_calibration"]["matrix"]
                source_height = self.data.cameras[frame.camera_id].height
                source_width = self.data.cameras[frame.camera_id].width
                photo_image = Image.fromarray(pixels)
                photo_image.thumbnail((640, 640))
                state.frustum = state.client.scene.add_camera_frustum(
                    "/selected_camera", fov=2 * np.arctan(source_height / (2 * k[1][1])),
                    aspect=source_width * k[1][1] / (source_height * k[0][0]),
                    scale=self.data.radius * .025, color=(255, 183, 65),
                    image=np.asarray(photo_image) if photo else None, format="png",
                    position=frame.center, wxyz=frame.wxyz, thickness=2., thickness_units="screen")
                state.points = state.client.scene.add_point_cloud(
                    "/selected_points", associated[::max(1, len(associated) // 5000)].astype(np.float32),
                    (0, 230, 255), point_size=self.data.radius * .003, precision="float32")
                state.png, state.filename = buffer.getvalue(), f"pose_{Path(frame.name).stem}_{'training' if space == 'Training input' else 'raw'}.png"
                if state.download:
                    state.client.send_file_download(state.filename, state.png)
                    state.download = False
                state.rendered = revision
            return True
        except (OSError, ValueError, RuntimeError) as error:
            with state.lock:
                state.controls["name"].value = frame.name
                state.controls["stats"].content = f"**Could not inspect image:** {error}"
                state.controls["image"].image = np.zeros((16, 24, 3), np.uint8)
                if state.frustum is not None:
                    state.frustum.visible = False
                if state.points is not None:
                    state.points.visible = False
                state.rendered = revision
                state.download = False
            return False

    def run(self):
        try:
            while not self.stopped.is_set():
                for state in list(self.clients.values()):
                    self.process(state)
                self.wake.wait(.1)
                self.wake.clear()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.wake.set()
        for state in list(self.clients.values()):
            state.connected = False
        self.server.stop()
