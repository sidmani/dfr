import torch
from .frustum import sphereToRect, buildFrustum, enumerateRays

# sampling schemes
def sampleRandom(near, far, count, device):
    rand = torch.sort(torch.rand(count, *near.shape, device=device), dim=0)[0]
    return ((far - near) * rand + near).permute(1, 2, 0)

def sampleUniform(near, far, count, device):
    # TODO: repeat/permute might be possible in 1 op
    divs = torch.linspace(0.0, 1.0, count, device=device).expand(*near.shape, count).permute(2, 0, 1)
    return ((far - near) * divs + near).permute(1, 2, 0)

def sampleStratified(near, far, count, device):
    n = float(count)
    divs = torch.linspace(0.0, 1.0, count, device=device).expand(*near.shape, count).permute(2, 0, 1)
    # TODO: double-check ranges
    rand = torch.rand(count, *near.shape) / (n - 1)
    return ((far - near) * (divs + rand) * (n - 1) / n + near).permute(1, 2, 0)

def scaleRays(rays, samples, cameraLoc):
    # [batch, selected #, sample count, 3]
    selectedRays = rays.unsqueeze(2).repeat(1, 1, samples.shape[1], 1)
    # [1, selected #, sample count, 1]
    selectedSamples = samples.view(1, -1, samples.shape[1], 1)

    # sum camera loc with scaled rays to compute sample points
    return cameraLoc.view(-1, 1, 1, 3) + selectedRays * selectedSamples
