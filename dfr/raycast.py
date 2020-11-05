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

def enumerateRays(phis, thetas, phiSpace, thetaSpace, cameraD):
    # x: the cartesian vector position of the camera
    x = cameraD * torch.stack([
        torch.cos(phis) * torch.sin(thetas),
        torch.sin(phis),
        torch.cos(phis) * torch.cos(thetas)
    ])

    phiBatch = torch.add(
                phis.view(-1, 1, 1),
                phiSpace.repeat(phis.shape[0], 1, 1))
    thetaBatch = torch.add(
                thetas.view(-1, 1, 1),
                thetaSpace.repeat(thetas.shape[0], 1, 1))

    cos_phi = torch.cos(phiBatch)
    rays = torch.transpose(
               torch.stack([
                   cos_phi * torch.sin(thetaBatch),
                   torch.sin(phiBatch),
                   cos_phi * torch.cos(thetaBatch),
               ]),
               (2, 1, 0))

# sampling schemes
def uniform_sample(s_1, s_2, count):
    divs = np.tile(np.linspace(0.0, 1.0, count, endpoint=False), (s_1.shape[0], 1))
    return (divs.T * (s_2 - s_1) + s_1).T

def random_sample(s_1, s_2, count):
    rands = np.random.rand(s_1.shape[0], count)
    return (rands.T * (s_2 - s_1) + s_1).T

def stratified_random_sample(s_1, s_2, count):
    divs = np.tile(np.linspace(0.0, 1.0, count, endpoint=False), (s_1.shape[0], 1))
    rands = np.random.rand(divs.shape) / float(count)
    return ((divs + rands).T * (s_2 - s_1) + s_1).T
