import numpy as np
import torch

def sphereToRect(phi, theta, r):
    cos_phi = torch.cos(phi)
    return r * torch.stack([
        cos_phi * torch.sin(theta),
        torch.sin(phi),
        cos_phi * torch.cos(theta)
    ], dim=-1)

class Frustum:
    def __init__(self, fov, px, device):
        # the location where the unit sphere would fill the fov
        self.cameraD = 1.0 / np.sin(fov / 2.0)
        self.imageSize = px
        self.fov = fov

        # theta increases counterclockwise in the zx plane
        # phi increases clockwise in the zy plane
        self.thetaSpace = np.pi + torch.linspace(
            fov / 2.0,
            -fov / 2.0,
            steps=px,
            device=device).repeat(px, 1)
        self.phiSpace = torch.transpose(torch.linspace(
            fov / 2.0,
            -fov / 2.0,
            steps=px,
            device=device).repeat(px, 1), 0, 1)

        # create a ray vector for each pixel
        self.viewField = sphereToRect(self.phiSpace, self.thetaSpace, 1.0).view(1, -1, 3, 1)

        center = self.cameraD * (torch.cos(self.phiSpace) - torch.cos(self.thetaSpace) - 1)

        # clamp negative values to 0 before sqrt
        radicand = torch.clamp(center ** 2 - self.cameraD ** 2 + 1.0, min=0.0)
        delta = torch.sqrt(radicand)

        # quadratic formula
        self.near = center - delta
        self.far = center + delta
        self.mask = (self.far - self.near) > 1e-10

    def jitteredViewField(self):
        device = self.near.device

        angle = self.fov / float(self.imageSize)
        jitterTheta = (torch.rand(*self.thetaSpace.shape, device=device) - 0.5) * angle
        jitterPhi = (torch.rand(*self.thetaSpace.shape, device=device) - 0.5) * angle

        newTheta = self.thetaSpace + jitterTheta
        newPhi = self.phiSpace + jitterPhi
        return sphereToRect(newPhi, newTheta, 1.0).view(1, -1, 3, 1)
