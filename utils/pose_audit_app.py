"""CPU pose audit: readable images and camera picking across the full trajectory."""

import base64
from dataclasses import dataclass, field
import html
import io
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import viser

from utils.pose_audit_data import COLORS
from utils.viewer_utils import overview_preset, quaternion_matrix


def pick_cameras(positions, wxyz, position, fov, width, height, screen_pos, radius=14):
    """Pick within a screen-pixel radius, independent of cloud or frustum meshes."""
    camera = (positions - position) @ quaternion_matrix(wxyz)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-8)
    projected = np.full((len(positions), 2), np.inf)
    focal = height / (2 * np.tan(fov / 2))
    projected[valid] = camera[valid, :2] / camera[valid, 2:3] * focal + [width / 2, height / 2]
    distance = np.linalg.norm(projected - np.asarray(screen_pos) * [width, height], axis=1)
    valid &= (distance <= radius)
    indices = np.flatnonzero(valid)
    # A depth tie-break makes coincident screen positions predictable.
    return indices[np.lexsort((camera[indices, 2], distance[indices]))][:20].tolist()


def image_html(png, filename, width, zoom="Fit"):
    """Keep the selected PNG at full resolution; native zoom scrolls, not resizes."""
    if not png:
        return '<p style="padding:12px">Select an image to inspect.</p>'
    size = "100%" if zoom == "Fit" else f"{width * (1 if zoom == '100%' else 2)}px"
    encoded = base64.b64encode(png).decode("ascii")
    return (f'<div style="max-height:68vh;overflow:auto;background:#10141b">'
            f'<img alt="{html.escape(filename, quote=True)}" src="data:image/png;base64,{encoded}" '
            f'style="display:block;width:{size};max-width:none;height:auto"></div>')


@dataclass
class AuditClient:
    client: object
    controls: dict = field(default_factory=dict)
    indices: list = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    lock: object = field(default_factory=threading.RLock)
    revision: int = 0
    rendered: int = -1
    connected: bool = True
    ready: bool = False
    download: bool = False
    find_matches: bool = False
    png: bytes = b""
    filename: str = "pose_overlay.png"
    image_width: int = 0
    frustum: object = None
    points: object = None
    cloud: object = None
    centers: object = None
    trajectory: object = None
    frustums: list = field(default_factory=list)
    panel: object = None


class _ClientGone(Exception):
    """Abort setup when a browser closes or the server stops mid-connection."""


class PoseAuditApp:
    def __init__(self, data, host="0.0.0.0", port=8081, max_points=100000, max_frustums=120):
        self.data, self.clients = data, {}
        self._initializing_clients = {}
        self._clients_lock = threading.RLock()
        self.wake, self.stopped = threading.Event(), threading.Event()
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(dark_mode=True, control_width="large")
        self.server.gui.main_panel.dock_right()
        self.server.gui.main_panel.set_width(330)
        self.server.scene.set_up_direction(data.up)
        self.chosen = np.linspace(0, len(data.xyz) - 1, min(len(data.xyz), max_points), dtype=int)
        self.frustum_indices = np.linspace(0, len(data.frames) - 1, min(max_frustums, len(data.frames)), dtype=int)
        self.server.scene.on_click()(self._pick)
        self.server.on_client_connect(self._connect)
        self.server.on_client_disconnect(self._disconnect)

    def _alive(self, state, require_ready=True):
        return (not self.stopped.is_set() and state.connected and (state.ready or not require_ready)
                and self.server.get_clients().get(state.client.client_id) is state.client)

    def _require_connection(self, state):
        if not self._alive(state, require_ready=False):
            raise _ClientGone()

    def _add_scene(self, state):
        self._require_connection(state)
        scene, data = state.client.scene, self.data
        state.cloud = scene.add_point_cloud(
            "/sfm_points", data.xyz[self.chosen].astype(np.float32), data.rgb[self.chosen],
            point_size=data.radius * .00025, point_shape="circle", point_shading="flat", precision="float32")
        state.centers = scene.add_point_cloud(
            "/camera_centers", data.positions.astype(np.float32),
            np.array([COLORS[frame.split] for frame in data.frames], np.uint8),
            point_size=data.radius * .0007, point_shape="circle", point_shading="flat", precision="float32")
        if len(data.frames) > 1:
            segments = np.stack((data.positions[:-1], data.positions[1:]), axis=1).astype(np.float32)
            state.trajectory = scene.add_line_segments("/trajectory", segments, (85, 118, 148),
                                                       thickness=1., thickness_units="screen")
        for index in self.frustum_indices:
            self._require_connection(state)
            frame = data.frames[index]
            camera = data.cameras[frame.camera_id]
            k = data.options[frame.camera_id]["colmap_calibration"]["matrix"]
            state.frustums.append(scene.add_camera_frustum(
                f"/cameras/{frame.id}", fov=2 * np.arctan(camera.height / (2 * k[1][1])),
                aspect=camera.width * k[1][1] / (camera.height * k[0][0]), scale=data.radius * .006,
                color=COLORS[frame.split], position=frame.center, wxyz=frame.wxyz,
                thickness=1., thickness_units="screen", visible=False))

    def _pick(self, event):
        with self._clients_lock:
            state = self.clients.get(event.client_id)
        if state is None or not self._alive(state):
            return
        camera = state.client.camera
        indices = pick_cameras(self.data.positions, camera.wxyz, camera.position, camera.fov,
                               camera.image_width, camera.image_height, event.screen_pos)
        if not indices:
            state.controls["pick_status"].content = "No camera within 14 pixels. Click a colored dot, or use Filename / Next image."
            return
        with state.lock:
            state.candidates = {f"{index + 1}: {self.data.frames[index].name}": index for index in indices}
            state.controls["nearby"].options = list(state.candidates)
            state.controls["nearby"].value = next(iter(state.candidates))
            state.controls["pick_status"].content = (f"{len(indices)} nearby camera(s). Use **Nearby cameras** when dots overlap. "
                                                     "Drag to orbit; scroll to zoom.")
        self.select_frame(event.client_id, indices[0])

    def _dirty(self, state):
        with state.lock:
            if not self._alive(state):
                return
            state.revision += 1
        self.wake.set()

    def _display(self, state):
        if not self._alive(state):
            return
        controls = state.controls
        state.cloud.visible = controls["cloud"].value
        state.cloud.point_size = self.data.radius * .00025 * controls["point_size"].value
        state.centers.point_size = self.data.radius * .0007 * controls["camera_size"].value
        for handle in state.frustums:
            handle.visible = controls["frustums"].value
        if state.trajectory is not None:
            state.trajectory.visible = controls["trajectory"].value
        if state.points is not None:
            state.points.visible = controls["matched"].value
        state.panel.visible = controls["inspector"].value

    def _refresh_image(self, state):
        with state.lock:
            if not self._alive(state):
                return
            state.controls["image"].content = image_html(state.png, state.filename, state.image_width,
                                                         state.controls["zoom"].value)

    def _overview(self, state, require_ready=True):
        if not self._alive(state, require_ready):
            return
        preset = overview_preset(self.data.center, self.data.radius, self.data.up)
        with state.client.atomic():
            state.client.camera.position = preset.position
            state.client.camera.wxyz = preset.wxyz
            state.client.camera.fov = preset.fov
            state.client.camera.look_at = self.data.center

    def _look(self, state):
        with state.lock:
            if not self._alive(state):
                return
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
        with self._clients_lock:
            if self.stopped.is_set():
                return
            self._initializing_clients[client.client_id] = state
        try:
            self._initialize(state)
            with state.lock:
                self._require_connection(state)
                self._overview(state, require_ready=False)
                # Publish only complete clients. Disconnect/stop removes the
                # pending entry, preventing late setup from resurrecting it.
                with self._clients_lock:
                    self._require_connection(state)
                    if self._initializing_clients.get(client.client_id) is not state:
                        return
                    state.ready = True
                    del self._initializing_clients[client.client_id]
                    self.clients[client.client_id] = state
                self._dirty(state)
        except _ClientGone:
            pass
        finally:
            with self._clients_lock:
                if self._initializing_clients.get(client.client_id) is state:
                    del self._initializing_clients[client.client_id]
            if not state.ready:
                state.connected = False

    def _initialize(self, state):
        client = state.client
        self._require_connection(state)
        groups = self.data.groups()
        default = "Training subset" if "Training subset" in groups else "All COLMAP images"
        state.indices = groups[default]
        self._add_scene(state)
        self._require_connection(state)
        gui, controls = client.gui, state.controls
        gui.add_markdown(f"## Image / pose audit\n{len(self.data.frames):,} cameras · {len(self.data.xyz):,} SfM points\n\n"
                         "This is the **original sparse COLMAP model**. Green: training · purple: test · gray: other cameras.")
        if self.data.root is None:
            gui.add_markdown("Dataset-only audit: no saved training subset or preprocessing configuration was supplied.")
        if not self.data.selection_known:
            gui.add_markdown("**Saved selection manifest missing:** training membership below represents candidates.")
        if not any(frame.count for frame in self.data.frames):
            gui.add_markdown("**No stored feature observations in this model.** Use Cloud projection for a coarse visual check; residual measurements are unavailable.")
        self._require_connection(state)
        with gui.add_folder("Choose image"):
            controls["group"] = gui.add_dropdown("Views", list(groups), initial_value=default)
            controls["index"] = gui.add_slider("Image index", 0, max(1, len(state.indices) - 1), 1, 0,
                                               disabled=len(state.indices) == 1)
            previous, following = gui.add_button("Previous image"), gui.add_button("Next image")
            controls["find"] = gui.add_text("Filename", "", hint="COLMAP filename or stem, e.g. 00961")
            find = gui.add_button("Find image")
            controls["name"] = gui.add_text("Selected image", "", disabled=True)
            controls["nearby"] = gui.add_dropdown("Nearby cameras", ["Click a camera dot"])
            controls["pick_status"] = gui.add_markdown("**Click any camera dot** in the 3D view to select it. Drag to orbit; scroll to zoom.")
            look, home = gui.add_button("Look from camera"), gui.add_button("Scene overview")
            find_matches = gui.add_button("Next image with matches")
        self._require_connection(state)
        with gui.add_folder("3D display", expand_by_default=False):
            controls["cloud"] = gui.add_checkbox("Show sparse cloud", True)
            controls["point_size"] = gui.add_slider("Point size", .25, 4., .25, 1.)
            controls["camera_size"] = gui.add_slider("Camera marker size", .25, 4., .25, 1.)
            controls["frustums"] = gui.add_checkbox("Show sampled frustums", False)
            controls["trajectory"] = gui.add_checkbox("Show trajectory", True)
            controls["matched"] = gui.add_checkbox("Highlight matched points", True)
            controls["photo"] = gui.add_checkbox("Image on selected frustum", False)
            controls["inspector"] = gui.add_checkbox("Show image inspector", True)
        with gui.add_folder("Measurements", expand_by_default=False):
            controls["stats"] = gui.add_markdown("Loading image…")

        self._require_connection(state)
        state.panel = gui.add_panel()
        with state.panel.add_tab("Image inspector"):
            controls["image_title"] = gui.add_markdown("Loading image…")
            controls["space"] = gui.add_dropdown("Image space", ["Training input", "Original image"],
                                                  initial_value="Training input" if self.data.root else "Original image")
            controls["mode"] = gui.add_dropdown("Marks", ["Both", "Observations", "Projections", "None", "Cloud projection"],
                                                 initial_value="Both")
            controls["zoom"] = gui.add_dropdown("Image zoom", ["Fit", "100%", "200%"], initial_value="Fit")
            controls["limit"] = gui.add_slider("Overlay points", 50, 2000, 50, 500)
            controls["alignment"] = gui.add_markdown("Loading alignment status…")
            controls["image"] = gui.add_html(image_html(b"", "", 0))
            download = gui.add_button("Download overlay PNG")
            gui.add_markdown("**Inspect:** select a frame → choose **100%** zoom and scroll the image → check whether cyan circles "
                             "and orange crosses coincide on real features. Red lines show their separation. "
                             "Resize or drag this panel to change the layout. Low residuals show internal SfM consistency; they do not rule out global drift or the wrong RGB file.")
        self._require_connection(state)
        state.panel.dock_left()
        state.panel.set_width(min(640, max(320, client.camera.image_width * .38)))

        @controls["group"].on_update
        def group_changed(_):
            with state.lock:
                previous_index = state.indices[min(int(controls["index"].value), len(state.indices) - 1)]
                state.indices = groups[controls["group"].value]
                controls["index"].max = max(1, len(state.indices) - 1)
                controls["index"].disabled = len(state.indices) == 1
                controls["index"].value = state.indices.index(previous_index) if previous_index in state.indices else 0
            self._dirty(state)

        for key in ("index", "space", "mode", "limit", "photo"):
            controls[key].on_update(lambda _, state=state: self._dirty(state))
        for key in ("cloud", "point_size", "camera_size", "frustums", "trajectory", "matched", "inspector"):
            controls[key].on_update(lambda _, state=state: self._display(state))
        controls["zoom"].on_update(lambda _: self._refresh_image(state))

        @controls["nearby"].on_update
        def nearby_changed(_):
            with state.lock:
                index = state.candidates.get(controls["nearby"].value)
            if index is not None:
                self.select_frame(client.client_id, index)

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
                controls["pick_status"].content = f"Found {len(matches)} matches. Enter a unique COLMAP filename."

        @find_matches.on_click
        def seek_matches(_):
            with state.lock:
                state.find_matches = True
                controls["pick_status"].content = "Checking feature records for another image with valid matches…"
            self._dirty(state)

        look.on_click(lambda _: self._look(state))
        home.on_click(lambda _: self._overview(state))

        @download.on_click
        def download_image(_):
            with state.lock:
                state.download = True
            self._dirty(state)

    def select_frame(self, client_id, index):
        with self._clients_lock:
            state = self.clients.get(client_id)
        if state is None:
            return
        controls = state.controls
        with state.lock:
            if not self._alive(state):
                return
            if index not in state.indices:
                controls["group"].value = "All COLMAP images"
                state.indices = self.data.groups()["All COLMAP images"]
                controls["index"].max = max(1, len(state.indices) - 1)
                controls["index"].disabled = len(state.indices) == 1
            controls["index"].value = state.indices.index(index)
        self._dirty(state)

    def _disconnect(self, client):
        with self._clients_lock:
            state = self.clients.pop(client.client_id, None)
            pending = self._initializing_clients.pop(client.client_id, None)
        state = state or pending
        if state:
            with state.lock:
                state.connected = state.ready = False
                state.png = b""
        self.wake.set()

    def process(self, state):
        with state.lock:
            if not self._alive(state) or state.revision == state.rendered:
                return False
            revision = state.revision
            index = state.indices[min(int(state.controls["index"].value), len(state.indices) - 1)]
            space, mode = state.controls["space"].value, state.controls["mode"].value
            limit, photo = int(state.controls["limit"].value), state.controls["photo"].value
            find_matches, indices = state.find_matches, state.indices.copy()
            state.find_matches = False
        frame = self.data.frames[index]
        try:
            if find_matches:
                target = self.data.next_with_matches(index, indices)
                with state.lock:
                    if not self._alive(state) or revision != state.revision:
                        return False
                    if target is not None:
                        self.select_frame(state.client.client_id, target)
                        state.controls["pick_status"].content = f"Found matched features: {self.data.frames[target].name}"
                        return False
                    state.controls["pick_status"].content = "No other image in the selected Views group has matching 3D point IDs. Try All COLMAP images, or Cloud projection."
            overlay, pixels, associated, details = self.data.inspect(index, space, limit, mode)
            buffer = io.BytesIO()
            Image.fromarray(overlay).save(buffer, format="PNG", compress_level=1)

            def metric(record):
                return "unavailable" if record["median"] is None else f"{record['median']:.3f} / {record['p95']:.3f} px"

            stats = (f"**{frame.split}** · RGB {details['source_size']} · training {details['training_size']}\n\n"
                     f"Stored features: **{details['feature_observations']:,}** · triangulated IDs: {details['triangulated_observations']:,}\n\n"
                     f"Associated points: **{details['associated_points']:,}** · valid training points: {details['valid_training_points']:,} · behind camera: {details['behind_camera']}\n\n"
                     f"COLMAP median / p95: **{metric(details['colmap_reprojection_px'])}** (original calibration size)\n\n"
                     f"Training median / p95: **{metric(details['training_projection_px'])}** (training pixels)\n\n"
                     f"Points inside native near cull (z ≤ 0.2 scene units): {details['within_native_near_cull']}")
            if "previous_translation" in details:
                stats += (f"\n\nPrevious-frame motion: {details['previous_translation']:.4g} SfM units · "
                          f"{details['previous_rotation_degrees']:.2f}°")
            if details["warnings"]:
                stats += "\n\n" + "\n\n".join(f"**Check:** {warning}" for warning in details["warnings"])
            if mode == "Cloud projection":
                alignment = (f"**Coarse cloud projection:** {details['cloud_overlay_points']:,} orange points. "
                             "Compare their structure with RGB edges. These points have no matched 2D observations or occlusion test; "
                             "they are not a reprojection-error measurement.")
            else:
                alignment = f"**{details['associated_points']:,} matched 3D points.** Cyan = stored features · orange = projections · red = residuals."
            if details["correspondence_status"]:
                alignment = ("**No matched points: alignment cannot be scored.** " + details["correspondence_status"] +
                             " Try **Next image with matches**, or set **Marks → Cloud projection** for a coarse visual check.\n\n" +
                             (alignment if mode == "Cloud projection" else "No correspondence marks are drawn on this image."))
            with state.lock:
                if not self._alive(state) or revision != state.revision:
                    return False
                state.controls["name"].value = frame.name
                state.controls["image_title"].content = f"**{frame.name}** · {details['training_size'][0]} × {details['training_size'][1]} pixels · {frame.split}"
                state.controls["stats"].content = stats
                state.controls["alignment"].content = alignment
                k = self.data.options[frame.camera_id]["colmap_calibration"]["matrix"]
                source_height = self.data.cameras[frame.camera_id].height
                source_width = self.data.cameras[frame.camera_id].width
                photo_image = Image.fromarray(pixels)
                photo_image.thumbnail((640, 640))
                state.frustum = state.client.scene.add_camera_frustum(
                    "/selected_camera", fov=2 * np.arctan(source_height / (2 * k[1][1])),
                    aspect=source_width * k[1][1] / (source_height * k[0][0]),
                    scale=self.data.radius * .018, color=(255, 183, 65),
                    image=np.asarray(photo_image) if photo else None, format="png",
                    position=frame.center, wxyz=frame.wxyz, thickness=2., thickness_units="screen")
                state.points = state.client.scene.add_point_cloud(
                    "/selected_points", associated[::max(1, len(associated) // 5000)].astype(np.float32),
                    (0, 230, 255), point_size=self.data.radius * .0005, point_shape="circle", point_shading="flat",
                    precision="float32", visible=state.controls["matched"].value)
                state.png, state.filename = buffer.getvalue(), f"pose_{Path(frame.name).stem}_{'training' if space == 'Training input' else 'raw'}.png"
                state.image_width = overlay.shape[1]
                self._refresh_image(state)
                if state.download:
                    state.client.send_file_download(state.filename, state.png)
                    state.download = False
                state.rendered = revision
            return True
        except (OSError, ValueError, RuntimeError) as error:
            with state.lock:
                if not self._alive(state) or revision != state.revision:
                    return False
                message = f"**Could not inspect image:** {error}"
                state.controls["name"].value = frame.name
                state.controls["image_title"].content = f"**{frame.name}**"
                state.controls["stats"].content = state.controls["alignment"].content = message
                state.png = b""
                self._refresh_image(state)
                if state.frustum is not None:
                    state.frustum.remove()
                    state.frustum = None
                if state.points is not None:
                    state.points.remove()
                    state.points = None
                state.rendered = revision
                state.download = False
            return False

    def run(self):
        try:
            while not self.stopped.is_set():
                with self._clients_lock:
                    states = list(self.clients.values())
                for state in states:
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
        with self._clients_lock:
            states = list(self.clients.values()) + list(self._initializing_clients.values())
            self.clients.clear()
            self._initializing_clients.clear()
        for state in states:
            with state.lock:
                state.connected = state.ready = False
                state.png = b""
        self.server.stop()
