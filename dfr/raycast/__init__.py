import torch
from torch.cuda.amp import autocast
import numpy as np
from .ray import rotateAxes, makeRays, multiscale

class MultiscaleFrustum:
    def __init__(self, fov, steps, device, dtype=torch.half):
        self.fov = fov
        self.cameraD = 1.0 / np.sin(fov / 2.0)
        self.edge = (self.cameraD - 1) * np.tan(fov / 2)

        self.scales = []
        self.slices = []
        self.near = []
        self.far = []
        self.mask = []

        with torch.no_grad():
            axes = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            z = axes[:, 2][:, None, None, :]

            size = 1
            for scale, slc in steps:
                size *= scale
                rays = makeRays(axes, size, self.cameraD, self.edge, dtype=dtype)
                cosines = (-z.unsqueeze(3) @ rays.unsqueeze(4)).view(-1, size, size)
                center = self.cameraD * cosines
                radicand = torch.clamp(center ** 2 - self.cameraD ** 2 + 1, min=0.0)
                delta = torch.sqrt(radicand)

                self.near.append(center - delta)
                self.far.append(center + delta)
                self.mask.append(delta > 1e-10)
                self.scales.append(scale)
                self.slices.append(slc)

        self.imageSize = size

    def __iter__(self):
        return zip(self.scales,
                   self.slices,
                   self.near,
                   self.far,
                   self.mask)

def raycast(phis, thetas, frustum, latents, sdf, gradScaler, threshold=5e-3):
    batch = latents.shape[0]
    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        axes = rotateAxes(phis, thetas)
        # run the raycasting in half precision
        critPoints, latents, sphereMask = multiscale(axes, frustum, latents, sdf, dtype=torch.float, threshold=threshold)
        critPoints = critPoints[sphereMask]
    critPoints.requires_grad = True

    with autocast():
        # sample the critical points with autograd enabled
        values, textures = sdf(critPoints, latents, sphereMask)
    del latents

    # compute normals
    # TODO: running gradScaler.scale() and gradScaler.getScale
    # forces a GPU-CPU sync, which is slow.
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
        unmaskedIllum = illum.clone()
        illum[notHitMask] = 1.0
        result = torch.zeros(batch, frustum.imageSize, frustum.imageSize, 4, device=phis.device)

        # shift the exponential over so that f(threshold) = 1, and clip anything to the left of that
        opacityMask = torch.exp(-10.0 * (values - threshold)).clamp(max=1.0)
        result[sphereMask] = torch.cat([illum * opacityMask * textures, opacityMask], dim=1)
        ret['image'] = result.permute(0, 3, 1, 2)
        ret['normals'] = normals
        ret['illum'] = unmaskedIllum

        return ret
