import numpy as np
import torch

def sphereToRect(phi, theta, r):
    cos_phi = torch.cos(phi)
    return r * torch.stack([
        cos_phi * torch.sin(theta),
        torch.sin(phi),
        cos_phi * torch.cos(theta)
    ], dim=-1)

def enumerateRays(phis, thetas, phiSpace, thetaSpace):
    # subtract because phiSpace and thetaSpace are mirrored over the camera plane
    phiBatch = phiSpace.repeat(phis.shape[0], 1, 1) - phis.view(-1, 1, 1)
    thetaBatch = thetaSpace.repeat(thetas.shape[0], 1, 1) - thetas.view(-1, 1, 1)

    return sphereToRect(phiBatch, thetaBatch, 1.0)

class Frustum:
    def __init__(self, fov, px, device):
        # D: the radial distance of the camera from the origin
        self.cameraD = 1.0 / np.sin(fov / 2.0)

        # theta increases counterclockwise in the zx plane
        # phi increases clockwise in the zy plane
        self.thetaSpace = torch.linspace(
            fov / 2.0,
            -fov / 2.0,
            steps=px,
            device=device).repeat(px, 1) + np.pi
        self.phiSpace = torch.transpose(torch.linspace(
            fov / 2.0,
            -fov / 2.0,
            steps=px,
            device=device).repeat(px, 1), 0, 1)

        # cos(theta - pi) = cos(pi - theta) = -cos(theta)
        center = self.cameraD * (torch.cos(self.phiSpace) - torch.cos(self.thetaSpace) - 1)

        # clamp negative values to 0 before sqrt
        radicand = torch.clamp(center ** 2 - self.cameraD ** 2 + 1.0, min=0.0)
        delta = torch.sqrt(radicand)

        # quadratic formula
        self.near = center - delta
        self.far = center + delta
        self.mask = (self.far - self.near) > 1e-10
