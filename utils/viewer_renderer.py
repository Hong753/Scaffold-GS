"""Native Scaffold-GS inference used by the browser viewer."""

import math
from types import SimpleNamespace

import numpy as np
import torch

from utils.viewer_utils import read_camera_presets, resolve_checkpoint, world_to_camera


def make_view_camera(wxyz, position, fov, width, height, appearance_id, device,
                     near=0.01, far=1000.0):
    from utils.graphics_utils import getProjectionMatrix

    fovx = 2 * math.atan(math.tan(fov / 2) * width / height)
    view = torch.as_tensor(world_to_camera(wxyz, position), device=device).transpose(0, 1)
    projection = getProjectionMatrix(near, far, fovx, fov).to(device).transpose(0, 1)
    return SimpleNamespace(
        uid=int(appearance_id), image_width=width, image_height=height, FoVx=fovx, FoVy=fov,
        znear=near, zfar=far, world_view_transform=view,
        full_proj_transform=view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
        camera_center=view.inverse()[3, :3],
    )


class ScaffoldRenderer:
    def __init__(self, model_path, iteration=-1, gpu=0):
        if not torch.cuda.is_available():
            raise RuntimeError("The native viewer needs CUDA and the compiled Scaffold-GS rasterizer. "
                               "Run it inside the scaffoldgs environment on your GPU machine.")
        torch.cuda.set_device(gpu)
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import prefilter_voxel, render

        self.root, self.checkpoint, native, config, self.iteration = resolve_checkpoint(model_path, iteration)
        self.config = config
        self.device = torch.device("cuda", gpu)
        settings = dict(feat_dim=32, n_offsets=10, voxel_size=0.001, update_depth=3,
                        update_init_factor=16, update_hierachy_factor=4, use_feat_bank=False,
                        appearance_dim=32, ratio=1, add_opacity_dist=False, add_cov_dist=False,
                        add_color_dist=False)
        settings.update({key: config[key] for key in settings if key in config})
        self.model = GaussianModel(**settings)
        self.model.load_ply_sparse_gaussian(str(native))
        self.model.load_mlp_checkpoints(str(self.checkpoint))
        self.model.eval()
        self.render_native, self.prefilter = render, prefilter_voxel
        self.pipeline = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        self.appearance_count = 0
        if self.model.appearance_dim:
            self.appearance_count = next(self.model.embedding_appearance.parameters()).shape[0]
        self.presets = read_camera_presets(self.root)
        self.anchor_count = self.model.get_anchor.shape[0]
        if not self.anchor_count:
            raise ValueError("Native checkpoint contains no anchors")
        points = self.model.get_anchor.detach()[::max(1, self.anchor_count // 8192)].cpu().numpy()
        lower, upper = np.quantile(points, [0.02, 0.98], axis=0)
        self.center = (lower + upper) / 2
        self.radius = max(float(np.linalg.norm(upper - lower) / 2), 1e-3)
        if self.presets:
            ups = np.array([-world_to_camera(p.wxyz, p.position)[:3, :3][1] for p in self.presets])
            self.up = ups.mean(axis=0)
            if np.linalg.norm(self.up) < 1e-6:
                self.up = ups[0]
        else:
            self.up = np.array([0., 0., 1.])
        self.up /= np.linalg.norm(self.up)
        self.default_background = (255, 255, 255) if config.get("white_background", False) else (0, 0, 0)

    @torch.inference_mode()
    def render(self, wxyz, position, fov, width, height, appearance_id=0,
               background=(0, 0, 0), scale=1.0):
        if self.appearance_count and not 0 <= appearance_id < self.appearance_count:
            raise ValueError("Appearance camera index is outside the saved embedding table")
        torch.cuda.set_device(self.device)
        camera = make_view_camera(wxyz, position, fov, width, height, appearance_id, self.device,
                                  far=max(1000., self.radius * 20))
        bg = torch.tensor(background, dtype=torch.float32, device=self.device) / 255
        visible = self.prefilter(camera, self.model, self.pipeline, bg, scaling_modifier=scale)
        if not visible.any():
            pixels = np.broadcast_to(np.asarray(background, dtype=np.uint8), (height, width, 3)).copy()
            return pixels, 0
        # The unchanged native renderer reevaluates opacity/color/covariance MLPs
        # for this exact camera. No single-view PLY baking or attribute averaging.
        package = self.render_native(camera, self.model, self.pipeline, bg,
                                     scaling_modifier=scale, visible_mask=visible, retain_grad=False)
        image = (package["render"].clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
        return image, len(package["radii"])
