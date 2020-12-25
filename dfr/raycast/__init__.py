import torch
from torch.cuda.amp import autocast
import numpy as np
from .ray import rotateAxes, makeRays, multiscale

def raycast(phis, thetas, scales, fov, latents, sdf, gradScaler, threshold=5e-3, sharpness=10.0):
    batch = latents.shape[0]
    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        axes = rotateAxes(phis, thetas)
        # run the raycasting in half precision
        critPoints, latents, sphereMask = multiscale(axes, scales, fov, latents, sdf, dtype=torch.float, threshold=threshold)
        critPoints = critPoints[sphereMask]
    critPoints.requires_grad = True

    with autocast():
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
    ret = {}
    with autocast():
        notHitMask = values > threshold
        # need epsilon in denominator for numerical stability
        normalLength = (normals.norm(dim=1).unsqueeze(1) + 1e-5)
        unitNormals = normals / normalLength

        # light is directed from camera
        light = axes[:, 2]

        # TODO: the indexing here is likely unnecessary and slows the backward pass
        # scale dot product from [-1, 1] to [0, 1]
        illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1) + 1.0) / 2.0
        illum[notHitMask] = 1.0
        imageSize = np.prod(scales)
        result = torch.zeros(batch, imageSize, imageSize, 4, device=phis.device)

        # shift the exponential over so that f(threshold) = 1, and clip anything to the left of that
        opacityMask = torch.exp(-sharpness * (values - threshold)).clamp(max=1.0)
        result[sphereMask] = torch.cat([illum * opacityMask * textures, opacityMask], dim=1)
        ret['image'] = result.permute(0, 3, 1, 2)
        ret['normals'] = normals

        return ret
