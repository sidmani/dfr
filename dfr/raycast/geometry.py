import torch
import numpy as np

# create rotation matrices from camera angles (phi, theta)
def rotateAxes(angles):
    phis, thetas = angles
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    cos_phi = torch.cos(phis)
    sin_phi = torch.sin(phis)

    # composition of 2 transforms: rotate theta first, then phi
    return torch.stack([
        torch.stack([cos_theta, -sin_theta * sin_phi, cos_phi * sin_theta], dim=1),
        torch.stack([torch.zeros_like(cos_phi), cos_phi, sin_phi], dim=1),
        torch.stack([-sin_theta, -sin_phi * cos_theta, cos_phi * cos_theta], dim=1),
    ], dim=2)

# construct a grid of rays viewing the origin from (0, 0, 1)
def rayGrid(axes, px, D, fov):
    # edge = (D - 1) * np.tan(fov / 2)

    # it's tempting to align the centers of the edge pixels, i.e.
    edge = (D - 1) * np.tan(fov / 2) * (1 - 1. / px)
    # but don't do it! the scale of the image changes across resolutions, which the discriminator can't handle

    xSpace = torch.linspace(-edge, edge, steps=px, device=axes.device).repeat(px, 1)[None, :, :, None]
    ySpace = -xSpace.transpose(1, 2)
    x = axes[:, 0][:, None, None, :]
    y = axes[:, 1][:, None, None, :]
    z = axes[:, 2][:, None, None, :]

    plane = xSpace * x + ySpace * y + z
    rays = plane - z * D
    norm = rays.reshape(-1, 3).norm(dim=1).view(-1, px, px, 1)
    return rays / norm

# find the near and far intersections of each ray with the unit sphere
def computePlanes(rays, axes, cameraD, size):
    z = axes[:, 2][:, None, None, :]
    center = cameraD * (-z.unsqueeze(3) @ rays.unsqueeze(4)).view(-1, size, size)
    # clamp to min=0, because rays that don't intersect the sphere have complex roots
    delta = (center ** 2 - cameraD ** 2 + 1).clamp(min=0.0).sqrt()
    return center - delta, center + delta, delta > 1e-5
