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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is not None:
        weights = mask.expand_as(img1)
        error = ((img1 - img2) ** 2) * weights
        mse = error.reshape(img1.shape[0], -1).sum(1, keepdim=True) / weights.reshape(
            img1.shape[0], -1).sum(1, keepdim=True).clamp_min(1)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
