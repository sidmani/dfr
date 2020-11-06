import numpy as np
import torch

def buildFrustum(fov, px, device=None):
    # D: the radial distance of the camera from the origin
    cameraD = 1.0 / np.sin(fov / 2.0)

    thetaSpace = torch.linspace(
        -fov / 2.0,
        fov / 2.0,
        steps=px,
        device=device).repeat(px, 1)
    phiSpace = torch.transpose(torch.flip(thetaSpace, [1]), 0, 1)

    center = cameraD * (torch.cos(phiSpace) + torch.cos(thetaSpace) - 1)

    # clamp negative values to 0 before sqrt
    radicand = torch.clamp(center ** 2 - cameraD ** 2 + 1.0, min=0.0)
    delta = torch.sqrt(radicand)

    segmentNear = center - delta
    segmentFar = center + delta

    return (cameraD, phiSpace, thetaSpace, segmentNear, segmentFar)

def sphereToRect(theta, phi, r):
    cos_phi = torch.cos(phi)
    return r * torch.stack([
        cos_phi * torch.sin(theta),
        torch.sin(phi),
        cos_phi * torch.cos(theta)
    ], dim=-1)

def enumerateRays(phis, thetas, phiSpace, thetaSpace):
    phiBatch = torch.add(
                phis.view(-1, 1, 1),
                phiSpace.repeat(phis.shape[0], 1, 1))
    thetaBatch = torch.add(
                thetas.view(-1, 1, 1),
                thetaSpace.repeat(thetas.shape[0], 1, 1))

    return sphereToRect(thetaBatch, phiBatch, 1.0)

# soft shading (DFR section 3.6)
def shader(values, k=10.0):
    return 1.0 / (1.0 + torch.exp(-k * values))

# sampling schemes
def sampleRandom(near, far, count, device):
    rand = torch.sort(torch.rand(count, *near.shape, device=device), dim=0)[0]
    return ((far - near) * rand + near).permute(1, 2, 0)

def sampleUniform(near, far, count, device):
    # TODO: repeat/permute might be possible in 1 op
    divs = torch.linspace(0.0, 1.0, count, device=device).repeat(*near.shape, 1).permute(2, 0, 1)
    return ((far - near) * divs + near).permute(1, 2, 0)

def sampleStratified(near, far, count, device):
    n = float(count)
    divs = torch.linspace(0.0, 1.0, count, device=device).repeat(*near.shape, 1).permute(2, 0, 1)
    rand = torch.rand(count, *near.shape) / (n - 1)
    return ((far - near) * (divs + rand) * (n - 1) / n + near).permute(1, 2, 0)
