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
    # TODO: also make small values (tangents) 0
    radicand = torch.clamp(center ** 2 - cameraD ** 2 + 1.0, min=0.0)
    delta = torch.sqrt(radicand)

    segmentNear = center - delta
    segmentFar = center + delta

    return (cameraD, phiSpace, thetaSpace, segmentNear, segmentFar)

def enumerateRays(phis, thetas, phiSpace, thetaSpace):
    phiBatch = torch.add(
                phis.view(-1, 1, 1),
                phiSpace.repeat(phis.shape[0], 1, 1))
    thetaBatch = torch.add(
                thetas.view(-1, 1, 1),
                thetaSpace.repeat(thetas.shape[0], 1, 1))

    cos_phi = torch.cos(phiBatch)
    return torch.stack([
               cos_phi * torch.sin(thetaBatch),
               torch.sin(phiBatch),
               cos_phi * torch.cos(thetaBatch),
           ], dim=3)

# sampling schemes
def sampleRandom(near, far, count):
    rand = torch.sort(torch.rand(count, *near.shape), dim=0)[0]
    return ((far - near) * rand + near).permute(1, 2, 0)

def sampleUniform(near, far, count):
    # TODO: repeat/permute might be possible in 1 op
    divs = torch.linspace(0.0, 1.0, count).repeat(*near.shape, 1).permute(2, 0, 1)
    return ((far - near) * divs + near).permute(1, 2, 0)

def sampleStratified(near, far, count):
    n = float(count)
    divs = torch.linspace(0.0, 1.0, count).repeat(*near.shape, 1).permute(2, 0, 1)
    rand = torch.rand(count, *near.shape) / (n - 1)
    return ((far - near) * (divs + rand) * (n - 1) / n + near).permute(1, 2, 0)
