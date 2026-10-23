"""Browser controls and latest-camera scheduling for native Scaffold rendering."""

from dataclasses import dataclass, field
import io
import logging
import threading
import time

from PIL import Image
import viser

from utils.viewer_utils import overview_preset, quaternion_matrix, render_dimensions


@dataclass
class ClientState:
    client: object
    controls: dict = field(default_factory=dict)
    lock: object = field(default_factory=threading.RLock)
    pose: object = None
    revision: int = 0
    moved_at: float = 0.0
    last_render: float = 0.0
    rendered_key: object = None
    download: bool = False
    connected: bool = True


class ScaffoldViewer:
    def __init__(self, renderer, host="0.0.0.0", port=8080, max_resolution=1600,
                 moving_resolution=800, fps=15):
        self.renderer = renderer
        self.max_resolution, self.moving_resolution, self.fps = max_resolution, moving_resolution, fps
        self.clients = {}
        self.clients_lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = False
        self.server = viser.ViserServer(host=host, port=port, label="Scaffold-GS Viewer")
        self.server.gui.configure_theme(dark_mode=True, show_logo=False,
                                        show_share_button=False, brand_color=(95, 160, 245))
        self.server.gui.main_panel.dock_right()
        self.server.scene.set_up_direction(tuple(renderer.up))
        self.overview = overview_preset(renderer.center, renderer.radius, renderer.up)
        self.presets = renderer.presets or [self.overview]
        self.server.on_client_connect(self._connect)
        self.server.on_client_disconnect(self._disconnect)

    def _invalidate(self, state, moving=False):
        with state.lock:
            state.revision += 1
            if moving:
                state.moved_at = time.monotonic()
        self.wake.set()

    def _camera_updated(self, state):
        camera = state.client.camera
        with state.lock:
            state.pose = (camera.wxyz.copy(), camera.position.copy(), float(camera.fov), float(camera.aspect))
        self._invalidate(state, moving=True)

    def _apply_preset(self, state, preset):
        rotation = quaternion_matrix(preset.wxyz)
        distance = max(self.renderer.radius * 0.25, 0.1)
        with state.client.atomic():
            state.client.camera.position = preset.position
            state.client.camera.up_direction = -rotation[:, 1]
            state.client.camera.look_at = preset.position + rotation[:, 2] * distance
            state.client.camera.fov = preset.fov
        state.controls["name"].value = preset.name
        if self.renderer.appearance_count:
            state.controls["appearance"].value = min(preset.appearance_id, self.renderer.appearance_count - 1)
        self._camera_updated(state)

    def _connect(self, client):
        state = ClientState(client)
        gui = client.gui
        gui.add_markdown(f"## Scaffold-GS\n**{self.renderer.root.name}** · iteration {self.renderer.iteration:,}\n\n"
                         f"{self.renderer.anchor_count:,} anchors · native rendering")
        with gui.add_folder("Camera"):
            state.controls["preset"] = gui.add_slider("Saved view", min=0, max=max(1, len(self.presets) - 1),
                                                      step=1, initial_value=0, disabled=len(self.presets) == 1)
            state.controls["name"] = gui.add_text("View name", self.presets[0].name, disabled=True)
            previous = gui.add_button("Previous view")
            following = gui.add_button("Next view")
            home = gui.add_button("Reset view")
            overview = gui.add_button("Scene overview")
            level = gui.add_button("Level horizon")
        with gui.add_folder("Render"):
            controls = state.controls
            controls["resolution"] = gui.add_slider("Stationary resolution", min=128, max=4096, step=16,
                                                       initial_value=self.max_resolution,
                                                       hint="Longest image side in pixels")
            controls["moving"] = gui.add_slider("Moving resolution", min=128, max=2048, step=16,
                                                   initial_value=min(self.moving_resolution, 2048))
            controls["quality"] = gui.add_slider("Moving JPEG quality", min=30, max=100, step=1, initial_value=85)
            controls["background"] = gui.add_rgb("Background", initial_value=self.renderer.default_background)
            controls["scale"] = gui.add_slider("Splat scale", min=0.1, max=2., step=0.05, initial_value=1.)
            controls["appearance"] = gui.add_slider("Appearance camera", min=0,
                                                       max=max(1, self.renderer.appearance_count - 1),
                                                       step=1, initial_value=0,
                                                       visible=bool(self.renderer.appearance_count))
            controls["pause"] = gui.add_checkbox("Pause rendering", initial_value=False)
            screenshot = gui.add_button("Download PNG")
            controls["status"] = gui.add_text("Status", "Waiting for camera", disabled=True)
        gui.add_markdown("**Navigate**\n\nDrag: orbit · right-drag: pan · wheel: zoom\n\n"
                         "W/A/S/D: fly · Q/E: down/up · arrow keys: look\n\n"
                         "Stationary frames and PNG downloads use lossless images.")

        @client.camera.on_update
        def camera_update(_):
            self._camera_updated(state)

        @controls["preset"].on_update
        def select_preset(_):
            self._apply_preset(state, self.presets[min(int(controls["preset"].value), len(self.presets) - 1)])

        def step_preset(offset):
            index = (int(controls["preset"].value) + offset) % len(self.presets)
            controls["preset"].value = index
            self._apply_preset(state, self.presets[index])

        previous.on_click(lambda _: step_preset(-1))
        following.on_click(lambda _: step_preset(1))
        home.on_click(lambda _: self._apply_preset(state, self.presets[int(controls["preset"].value)]))
        overview.on_click(lambda _: self._apply_preset(state, self.overview))

        @level.on_click
        def level_horizon(_):
            client.camera.up_direction = self.renderer.up
            self._camera_updated(state)

        @screenshot.on_click
        def download(_):
            with state.lock:
                state.download = True
            self._invalidate(state)

        for key in ("resolution", "moving", "quality", "background", "scale", "appearance", "pause"):
            controls[key].on_update(lambda _: self._invalidate(state))
        with self.clients_lock:
            self.clients[client.client_id] = state
        self._apply_preset(state, self.presets[0])

    def _disconnect(self, client):
        with self.clients_lock:
            state = self.clients.pop(client.client_id, None)
        if state is not None:
            state.connected = False

    def render_pending(self, state, now=None):
        """Process only the newest camera. Called serially on the CUDA thread."""
        now = time.monotonic() if now is None else now
        with state.lock:
            if not state.connected or state.pose is None or (state.controls["pause"].value and not state.download):
                return False
            if now - state.last_render < 1 / self.fps and not state.download:
                return False
            revision, pose, download = state.revision, state.pose, state.download
            moving = now - state.moved_at < 0.2 and not download
            maximum = int(state.controls["resolution"].value)
            size = min(maximum, int(state.controls["moving"].value)) if moving else maximum
            wxyz, position, fov, aspect = pose
            width, height = render_dimensions(aspect, size)
            key = (revision, width, height, moving)
            if key == state.rendered_key and not download:
                return False
            appearance_id = int(state.controls["appearance"].value)
            background = tuple(state.controls["background"].value)
            scale = float(state.controls["scale"].value)
            quality = int(state.controls["quality"].value)
        start = time.perf_counter()
        try:
            image, count = self.renderer.render(wxyz, position, fov, width, height,
                                                appearance_id=appearance_id, background=background, scale=scale)
            elapsed = time.perf_counter() - start
            with state.lock:
                state.last_render = now
                if not state.connected or (download and state.revision != revision):
                    return False
                # Show completed interactive frames even if the camera moved
                # during inference. Discarding every such frame would freeze
                # the display whenever mouse events arrive faster than CUDA.
                # The next inference uses the latest pose, with no pose queue.
                state.client.scene.set_background_image(image, format="jpeg" if moving else "png",
                                                        jpeg_quality=quality if moving else None)
                state.rendered_key = key
                state.controls["status"].value = (f"{width}×{height} · {elapsed * 1000:.0f} ms · "
                                                    f"{count:,} child Gaussians")
                if download:
                    stream = io.BytesIO()
                    Image.fromarray(image).save(stream, format="PNG")
                    state.client.send_file_download(f"scaffold_{self.renderer.iteration}.png", stream.getvalue())
                    state.download = False
            return True
        except Exception as error:
            logging.exception("Native viewer render failed")
            with state.lock:
                state.last_render, state.rendered_key, state.download = now, key, False
                state.controls["status"].value = f"Render error: {error}. Try a lower resolution."
            return False

    def run(self):
        while not self.closed:
            with self.clients_lock:
                states = list(self.clients.values())
            for state in states:
                self.render_pending(state)
            self.wake.wait(0.01)
            self.wake.clear()

    def stop(self):
        if not self.closed:
            self.closed = True
            with self.clients_lock:
                for state in self.clients.values():
                    with state.lock:
                        state.connected = False
            self.wake.set()
            self.server.stop()
