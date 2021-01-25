import torch

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

# construct a grid of rays viewing the origin from the camera
def rayGrid(axes, px, D):
    edge = 1. - 1 / px
    xSpace = torch.linspace(-edge, edge, steps=px, device=axes.device).repeat(px, 1)[None, :, :, None]
    ySpace = -xSpace.transpose(1, 2)
    x = axes[:, 0][:, None, None, :]
    y = axes[:, 1][:, None, None, :]
    z = axes[:, 2][:, None, None, :]

    rays = xSpace * x + ySpace * y - z * D
    return rays / rays.norm(dim=3, keepdim=True)

# find the near and far intersections of each ray with the unit sphere
# the solution is a quadratic in the distance along the ray with complex roots iff the ray misses
def computePlanes(rays, axes, cameraD):
    z = axes[:, 2][:, None, None, None, :]
    center = cameraD * (-z @ rays.unsqueeze(4)).flatten(2)
    # clamp to min=0 to ignore complex roots
    delta = (center ** 2 - cameraD ** 2 + 1).clamp(min=0.).sqrt()
    return center - delta, center + delta, delta > 1e-5
