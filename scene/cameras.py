#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.image_loading import load_image_tensor

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_path=None, image_resolution=None, image_options=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_path = image_path
        self.image_options = image_options or {}
        self._image = None
        if image_path is not None:
            if image_resolution is None:
                raise ValueError("Disk-backed cameras require image_resolution")
            self.image_width, self.image_height = map(int, image_resolution)
            if self.image_width < 1 or self.image_height < 1:
                raise ValueError("Camera image dimensions must be positive")
        else:
            # Retain the explicit tensor-constructor API for external callers.
            # Dataset cameras always supply a path and never enter this branch.
            if image is None:
                raise ValueError("Camera requires an image_path or an image tensor")
            self._image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self._image.shape[2]
            self.image_height = self._image.shape[1]
            if gt_alpha_mask is not None:
                self._image *= gt_alpha_mask.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load_image(self, device=None):
        """Load the selected view onto device, without storing pixels on Camera."""
        target = self.data_device if device is None else device
        if self.image_path is None:
            return self._image.to(target)
        image = load_image_tensor(
            self.image_path, (self.image_width, self.image_height), self.image_options)
        return image.to(target)

    @property
    def original_image(self):
        # Compatibility for external callers; this property also reads on demand.
        return self.load_image()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

