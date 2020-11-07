import numpy as np
import torch

def sphereToRect(theta, phi, r):
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
    segmentNear = center - delta
    segmentFar = center + delta

    return (cameraD, phiSpace, thetaSpace, segmentNear, segmentFar)

def enumerateRays(phis, thetas, phiSpace, thetaSpace):
    # subtract because phiSpace and thetaSpace are mirrored over the camera plane
    phiBatch = phiSpace.repeat(phis.shape[0], 1, 1) - phis.view(-1, 1, 1)
    thetaBatch = thetaSpace.repeat(thetas.shape[0], 1, 1) - thetas.view(-1, 1, 1)

    return sphereToRect(thetaBatch, phiBatch, 1.0)
