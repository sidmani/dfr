import torch
from torch.cuda.amp import autocast
from collections import namedtuple
import numpy as np
from .geometry import rotateAxes
from .ray import multiscale
from .lighting import shade
from ..flags import Flags

SurfaceData = namedtuple('SurfaceData', ['values', 'textures', 'normals', 'normalLength'])

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

def evalSurface(data, sdf, gradScaler, threshold):
    with autocast(enabled=Flags.AMP):
        # sample the critical points with autograd enabled
        # TODO: latent masking in sdf forward doesn't need grad
        values, textures = sdf(data.points, data.latents, data.mask)

    # compute normals
    scaledNormals = torch.autograd.grad(outputs=gradScaler.scale(values),
                inputs=data.points,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
    normals = scaledNormals / gradScaler.get_scale()

    normalLength = normals.norm(dim=1, keepdim=True)
    # need epsilon in denominator for numerical stability
    unitNormals = normals / (normalLength + 1e-5)

    return SurfaceData(values - threshold, textures, unitNormals, normalLength)

def raycast(angles, scales, latents, sdf, gradScaler, sharpness, halfSharpness=None, threshold=5e-3):
    with torch.no_grad():
        axes = rotateAxes(angles)
        fullData, halfData = multiscale(axes, scales, latents, sdf, threshold, half=halfSharpness is not None)

    fullSurface = evalSurface(fullData, sdf, gradScaler, threshold)
    if halfSharpness is not None:
        halfSurface = evalSurface(halfData, sdf, gradScaler, threshold)

    with autocast(enabled=Flags.AMP):
        # light is directed from the point below the camera on the zx plane
        light = axes[:, 2]
        # project to zx plane and normalize
        # light[:, 1] = 0
        # light = light / light.norm(dim=1).unsqueeze(1)
        ret = {'normalLength': fullSurface.normalLength}
        ret['full'] = shade(fullSurface, light, fullData.mask, sharpness)
        if halfSharpness is not None:
            ret['half'] = shade(halfSurface, light, halfData.mask, halfSharpness)

        return ret
