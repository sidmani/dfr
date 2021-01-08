import torch
import numpy as np

# create rotation matrices from camera angles (phi, theta)
def rotateAxes(angles):
    phis, thetas = angles
    cos_theta = torch.cos(thetas).unsqueeze(1)
    sin_theta = torch.sin(thetas).unsqueeze(1)
    cos_phi = torch.cos(phis).unsqueeze(1)
    sin_phi = torch.sin(phis).unsqueeze(1)

    # composition of 2 transforms: rotate theta first, then phi
    return torch.cat([
        torch.cat([cos_theta, -sin_theta * sin_phi, cos_phi * sin_theta], dim=1).unsqueeze(2),
        torch.cat([torch.zeros_like(cos_phi), cos_phi, sin_phi], dim=1).unsqueeze(2),
        torch.cat([-sin_theta, -sin_phi * cos_theta, cos_phi * cos_theta], dim=1).unsqueeze(2),
    ], dim=2)

# construct a grid of rays viewing the origin from (0, 0, 1)
def rayGrid(axes, px, D, fov, dtype):
    edgeLength = (D - 1) * np.tan(fov / 2)

    # offsets the edges so a 2n x 2n grid is evenly spaced within an n x n grid
    # for example, raycasting at 32x32 avg pooled to 16x16 should look very similar to raycasting at 16x16
    edge = edgeLength * (1. - 1. / px)

    xSpace = torch.linspace(-edge, edge, steps=px, dtype=dtype, device=axes.device).repeat(px, 1)[None, :, :, None]
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
    delta = torch.sqrt(torch.clamp(center ** 2 - cameraD ** 2 + 1, min=0.0))
    return center - delta, center + delta, delta > 1e-10
