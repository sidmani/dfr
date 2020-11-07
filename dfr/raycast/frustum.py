import numpy as np
import torch

def sphereToRect(phi, theta, r):
    cos_phi = torch.cos(phi)
    return r * torch.stack([
        cos_phi * torch.sin(theta),
        torch.sin(phi),
        cos_phi * torch.cos(theta)
    ], dim=-1)

def buildFrustum(fov, px, device):
    # D: the radial distance of the camera from the origin
    cameraD = 1.0 / np.sin(fov / 2.0)

    # theta increases counterclockwise in the zx plane
    # phi increases clockwise in the zy plane
    thetaSpace = torch.linspace(
        fov / 2.0,
        -fov / 2.0,
        steps=px,
        device=device).repeat(px, 1) + np.pi
    phiSpace = torch.transpose(torch.linspace(
        fov / 2.0,
        -fov / 2.0,
        steps=px,
        device=device).repeat(px, 1), 0, 1)

    # cos(theta - pi) = cos(pi - theta) = -cos(theta)
    center = cameraD * (torch.cos(phiSpace) - torch.cos(thetaSpace) - 1)

    # clamp negative values to 0 before sqrt
    radicand = torch.clamp(center ** 2 - cameraD ** 2 + 1.0, min=0.0)
    delta = torch.sqrt(radicand)

    # quadratic formula
    near = center - delta
    far = center + delta

    return Frustum(cameraD, phiSpace, thetaSpace, near, far)

def enumerateRays(phis, thetas, phiSpace, thetaSpace):
    # subtract because phiSpace and thetaSpace are mirrored over the camera plane
    phiBatch = phiSpace.repeat(phis.shape[0], 1, 1) - phis.view(-1, 1, 1)
    thetaBatch = thetaSpace.repeat(thetas.shape[0], 1, 1) - thetas.view(-1, 1, 1)

    return sphereToRect(phiBatch, thetaBatch, 1.0)

class Frustum:
    def __init__(self, cameraD, phiSpace, thetaSpace, near, far):
        self.cameraD = cameraD
        self.phiSpace = phiSpace
        self.thetaSpace = thetaSpace
        self.near = near
        self.far = far
        self.mask = (far - near) > 1e-10
