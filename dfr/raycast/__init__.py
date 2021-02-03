import torch
from torch.cuda.amp import autocast
from collections import namedtuple
import numpy as np
from .geometry import rotateAxes
from .ray import multiscale
from .lighting import shade
from ..flags import Flags

SurfaceData = namedtuple('SurfaceData', ['values', 'textures', 'normals', 'normalLength'])

def sample(batchSize, device, ckpt, scales, sigma):
  # elevation is between 20 and 30 deg (per dataset)
  deg20 = 20 * np.pi / 180
  deg10 = 10 * np.pi / 180
  phis = torch.rand(batchSize, device=device) * deg10 + deg20
  # azimuth is uniform in [0, 2pi]
  # thetas = torch.rand_like(phis) * (2.0 * np.pi)
  phis = torch.ones(batchSize, device=device) * 25 * np.pi / 180
  thetas = torch.zeros_like(phis)
  angles = (phis, thetas)
  z = torch.normal(0.0, ckpt.hparams.latentStd, (batchSize, ckpt.hparams.latentSize), device=device)
  return raycast(angles, scales, z, ckpt.gen, ckpt.gradScaler, sigma)

def evalSurface(data, sdf, gradScaler, threshold):
  with autocast(enabled=Flags.AMP):
    # sample the critical points with autograd enabled
    # TODO: latent masking in sdf forward doesn't need grad
    # TODO: latents are recomputed even though there are repeats
    values, textures = sdf(data.points, data.latents, data.mask)

  # compute normals
  scaledNormals = torch.autograd.grad(outputs=gradScaler.scale(values),
        inputs=data.points,
        grad_outputs=torch.ones_like(values),
        create_graph=True)[0]
  normals = scaledNormals / gradScaler.get_scale()

  normalLength = normals.norm(dim=1, keepdim=True)
  # need epsilon in denominator for numerical stability
  unitNormals = normals / (normalLength + 1e-5)

  # values are shifted so the threshold is at 0, and surface < 0
  return SurfaceData(values - threshold, textures, unitNormals, normalLength)

def raycast(angles, scales, latents, sdf, gradScaler, sigma, threshold=5e-3):
  with torch.no_grad():
    axes = rotateAxes(angles)
    fullData = multiscale(axes, scales, latents, sdf, threshold)

  with autocast(enabled=Flags.AMP):
    fullSurface = evalSurface(fullData, sdf, gradScaler, threshold)
    # light is directed from the point below the camera on the zx plane
    light = axes[:, 2]
    # project to zx plane and normalize
    # light[:, 1] = 0
    # light = light / light.norm(dim=1).unsqueeze(1)
    return {
      'normalLength': fullSurface.normalLength,
      'full': shade(fullSurface, light, fullData.mask, sigma)
    }
