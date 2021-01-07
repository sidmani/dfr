import torch
from torch.cuda.amp import autocast
import numpy as np
from .ray import rotateAxes, multiscale
from ..flags import Flags
from ..util import normalizedZ

def sample_like(other, ckpt, scales, sharpness):
    batchSize = other.shape[0]
    device = other.device
    return sample(other.shape[0], other.device, ckpt, scales, sharpness)

def sample(batchSize, device, ckpt, scales, sharpness):
    # elevation is between 20 and 30 deg (per dataset)
    deg20 = 20 * np.pi / 180
    deg10 = 10 * np.pi / 180
    phis = torch.rand(batchSize, device=device) * deg10 + deg20
    # azimuth is uniform in [0, 2pi]
    thetas = torch.rand_like(phis) * (2.0 * np.pi)
    z = normalizedZ((batchSize, ckpt.hparams.latentSize), device)
    return raycast(phis, thetas, scales, ckpt.hparams.fov, z, ckpt.gen, ckpt.gradScaler, sharpness)

def raycast(phis, thetas, scales, fov, latents, sdf, gradScaler, sharpness, threshold=5e-3):
    batch = latents.shape[0]
    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        axes = rotateAxes(phis, thetas)
        # run the raycasting in half precision
        critPoints, latents, sphereMask = multiscale(axes, scales, fov, latents, sdf, dtype=torch.float, threshold=threshold)
        critPoints = critPoints[sphereMask]
    critPoints.requires_grad = True

    with autocast(enabled=Flags.AMP):
        # sample the critical points with autograd enabled
        values, textures = sdf(critPoints, latents, sphereMask)
    del latents

    # compute normals
    # TODO: running gradScaler.scale() and gradScaler.getScale forces a GPU-CPU sync
    scaledNormals = torch.autograd.grad(outputs=gradScaler.scale(values),
                inputs=critPoints,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
    normals = scaledNormals / gradScaler.get_scale()

    with autocast(enabled=Flags.AMP):
        notHitMask = values > threshold
        normalLength = normals.norm(dim=1)
        # need epsilon in denominator for numerical stability
        unitNormals = normals / (normalLength.unsqueeze(1) + 1e-5)

        # light is directed from the point below the camera on the zx plane
        light = axes[:, 2]
        # project to zx plane and normalize
        # light[:, 1] = 0
        # light = light / light.norm(dim=1).unsqueeze(1)

        # TODO: the indexing here is likely unnecessary and slows the backward pass
        # scale dot product from [-1, 1] to [0, 1]
        illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1) + 1.0) / 2.0
        illum[notHitMask] = 1.0
        imageSize = np.prod(scales)

        # shift the exponential over so that f(threshold) = 1, and clip anything to the left of that
        opacityMask = torch.exp(-sharpness * (values - threshold)).clamp(max=1.0)
        # opacityMask = torch.nn.functional.threshold(opacityMask, 1e-3, 0.)
        result = torch.zeros(batch, imageSize, imageSize, 4, device=phis.device)
        result[sphereMask] = torch.cat([illum * opacityMask * textures, opacityMask], dim=1)

        # TODO: this permute() can be avoided
        return {'image': result.permute(0, 3, 1, 2), 'normalLength': normalLength}
