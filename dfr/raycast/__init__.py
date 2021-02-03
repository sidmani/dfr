import torch
from torch.cuda.amp import autocast
import numpy as np
from .geometry import rotateAxes
from .ray import multiscale
from .lighting import shade
from ..flags import Flags
from ..util import grad, normalize

def raycast(angles, latents, scales, sdf, gradScaler, threshold=5e-3):
  with torch.no_grad():
    axes = rotateAxes(angles)
    data = multiscale(axes, scales, latents, sdf, threshold)

  with autocast(enabled=Flags.AMP):
    # sample the critical points with autograd enabled
    # TODO: latent masking in sdf forward doesn't need grad
    # TODO: latents are recomputed even though there are repeats
    values, textures = sdf(data.points, data.latents, data.mask)

  # compute normals
  normals = grad(data.points, values, gradScaler)

  with autocast(enabled=Flags.AMP):
    unitNormals, length = normalize(normals, dim=1, keepLength=True, eps=1e-5)
    # fullSurface = evalSurface(data, sdf, gradScaler, threshold)
    # light is directed from the point below the camera on the zx plane
    light = axes[:, 2]
    # project to zx plane and normalize
    # light[:, 1] = 0
    # light = light / light.norm(dim=1).unsqueeze(1)
    image = shade(values - threshold, textures, unitNormals, light, data.mask)

    return image, length
