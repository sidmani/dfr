import torch
import numpy as np
from .ray import rotateAxes, makeRays, multiscale

class MultiscaleFrustum:
    def __init__(self, fov, steps, device):
        axes = torch.eye(3, device=device).unsqueeze(0)
        z = axes[:, 2][:, None, None, :]

        self.fov = fov
        self.cameraD = 1.0 / np.sin(fov / 2.0)
        self.edge = (self.cameraD - 1) * np.tan(fov / 2)

        self.scales = []
        self.slices = []
        self.near = []
        self.far = []
        self.mask = []

        size = 1
        for scale, slc in steps:
            size *= scale
            rays = makeRays(axes, size, self.cameraD, self.edge)
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

def raycast(phis, thetas, frustum, latents, sdf):
    batch = latents.shape[0]
    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        axes = rotateAxes(phis, thetas)
        critPoints, latents, sphereMask = multiscale(axes, frustum, latents, sdf)
    critPoints.requires_grad = True

    # sample the critical points with autograd enabled
    values, textures = sdf(critPoints, latents)

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    # need epsilon in denominator for numerical stability
    unitNormals = normals / (normals.norm(dim=1).unsqueeze(1) + 1e-5)

    # light is directed from camera
    light = axes[:, 2]
    notHitMask = values > 0.0

    # scale dot product from [-1, 1] to [0, 1]
    illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1) + 1.0) / 2.0
    illum[notHitMask] = 1.0

    result = torch.zeros(batch, frustum.imageSize, frustum.imageSize, 4, device=phis.device)
    opacityMask = torch.ones_like(values)
    opacityMask[notHitMask] = torch.exp(-10.0 * values[notHitMask])
    result[sphereMask] = torch.cat([opacityMask * illum * textures, opacityMask], dim=1)

    return result.permute(0, 3, 1, 2), normals
