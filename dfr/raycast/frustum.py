import numpy as np
import torch

def sphereToRect(phi, theta, r):
    cos_phi = torch.cos(phi)
    return r * torch.stack([
        cos_phi * torch.sin(theta),
        torch.sin(phi),
        cos_phi * torch.cos(theta)
    ], dim=-1)

def enumerateRays(phis, thetas, viewField, imageSize):
    device = phis.device
    # [px, px, 3, 3]
    zeros = torch.zeros(phis.shape[0], device=device)
    ones = torch.ones(phis.shape[0], device=device)
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    cos_phi = torch.cos(phis)
    sin_phi = torch.sin(phis)

    rotation = torch.stack([
        torch.stack([cos_theta, -sin_theta * sin_phi, sin_theta * cos_phi]),
        torch.stack([zeros, cos_phi, sin_phi]),
        torch.stack([-sin_theta, -sin_phi * cos_theta, cos_phi * cos_theta]),
    ]).permute(2, 0, 1).unsqueeze(1)

    return torch.matmul(rotation, viewField).view(-1, imageSize, imageSize, 3)

class Frustum:
    def __init__(self, fov, px, device):
        # the location where the unit sphere would fill the fov
        idealCameraLoc = 1.0 / np.sin(fov / 2.0)
        # self.cameraD = idealCameraLoc

        # actual camera location is farther away, to avoid fisheye distortion
        self.cameraD = 1.0 / np.sin(fov / 2.5)

        # theta increases counterclockwise in the zx plane
        # phi increases clockwise in the zy plane
        thetaSpace = np.pi + torch.linspace(
            fov / 2.0,
            -fov / 2.0,
            steps=px,
            device=device).repeat(px, 1)
        phiSpace = torch.transpose(torch.linspace(
            fov / 2.0,
            -fov / 2.0,
            steps=px,
            device=device).repeat(px, 1), 0, 1)

        # create a ray vector for each pixel
        self.viewField = sphereToRect(phiSpace, thetaSpace, 1.0).view(1, -1, 3, 1)

        center = idealCameraLoc * (torch.cos(phiSpace) - torch.cos(thetaSpace) - 1)

        # clamp negative values to 0 before sqrt
        radicand = torch.clamp(center ** 2 - idealCameraLoc ** 2 + 1.0, min=0.0)
        delta = torch.sqrt(radicand)

        # quadratic formula
        self.near = center - delta
        self.far = center + delta
        self.mask = (self.far - self.near) > 1e-10
