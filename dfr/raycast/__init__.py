import torch
from torch.cuda.amp import autocast
from collections import namedtuple
import numpy as np
from .geometry import rotateAxes
from .ray import multiscale
from .lighting import illuminate, fuzz
from ..flags import Flags

def sample_like(other, ckpt, scales, sharpness, halfSharpness=None):
    batchSize = other.shape[0]
    device = other.device
    return sample(other.shape[0], other.device, ckpt, scales, sharpness, halfSharpness)

def sample(batchSize, device, ckpt, scales, sharpness, halfSharpness=None):
    # elevation is between 20 and 30 deg (per dataset)
    deg20 = 20 * np.pi / 180
    deg10 = 10 * np.pi / 180
    phis = torch.rand(batchSize, device=device) * deg10 + deg20
    # azimuth is uniform in [0, 2pi]
    thetas = torch.rand_like(phis) * (2.0 * np.pi)
    angles = (phis, thetas)
    z = torch.normal(0.0, ckpt.hparams.latentStd, (batchSize, ckpt.hparams.latentSize), device=device)
    return raycast(angles, scales, z, ckpt.gen, ckpt.gradScaler, sharpness, halfSharpness)

def raycast(angles, scales, latents, sdf, gradScaler, sharpness, halfSharpness=None, threshold=5e-3):
    with torch.no_grad():
        axes = rotateAxes(angles)
        critPoints, latents, sphereMask = multiscale(axes, scales, latents, sdf, dtype=torch.float, threshold=threshold)
        critPoints = critPoints[sphereMask]
    critPoints.requires_grad = True

    with autocast(enabled=Flags.AMP):
        # sample the critical points with autograd enabled
        # TODO: latent masking in sdf forward doesn't need grad
        values, textures = sdf(critPoints, latents, sphereMask)
    del latents

    # compute normals
    scaledNormals = torch.autograd.grad(outputs=gradScaler.scale(values),
                inputs=critPoints,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
    normals = scaledNormals / gradScaler.get_scale()

    # shade the result
    with autocast(enabled=Flags.AMP):
        normalLength = normals.norm(dim=1, keepdim=True)
        # need epsilon in denominator for numerical stability
        unitNormals = normals / (normalLength + 1e-5)

        # light is directed from the point below the camera on the zx plane
        light = axes[:, 2]
        # project to zx plane and normalize
        # light[:, 1] = 0
        # light = light / light.norm(dim=1).unsqueeze(1)

        shading = torch.zeros(*sphereMask.shape, 3, device=sphereMask.device)
        illum = illuminate(axes[:, 2], unitNormals)
        illum[values > threshold] = 1.0
        shading[sphereMask] = (illum * textures).float()

        # opacityMask = torch.nn.functional.threshold(opacityMask, 1e-3, 0.)
        valueMap = torch.ones(*sphereMask.shape, 1, device=sphereMask.device)
        valueMap[sphereMask] = values.float()

        valueMap = valueMap.permute(0, 3, 1, 2)
        shading = shading.permute(0, 3, 1, 2)

        ret = {'normalLength': normalLength}
        ret['image'] = composite(valueMap, shading, threshold, sharpness)

        if halfSharpness is not None:
            valueMapHalf = torch.nn.functional.interpolate(valueMap, scale_factor=0.5, mode='bilinear')
            shadingHalf = torch.nn.functional.interpolate(shading, scale_factor=0.5, mode='bilinear')
            ret['half'] = composite(valueMapHalf, shadingHalf, threshold, halfSharpness)

        return ret

def composite(valueMap, shading, threshold, sharpness):
    opacity = fuzz(valueMap - threshold, sharpness)
    return torch.cat([shading * opacity, opacity], dim=1)
