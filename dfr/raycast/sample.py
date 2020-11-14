import torch

# sampling schemes
def sampleRandom(near, far, count):
    device = near.device

    rand = torch.sort(torch.rand(count, *near.shape, device=device), dim=0)[0]
    return ((far - near) * rand + near).permute(1, 2, 0)

def sampleUniform(near, far, count):
    device = near.device

    # TODO: repeat/permute might be possible in 1 op
    divs = torch.linspace(0.0, 1.0, count, device=device).expand(*near.shape, count).permute(2, 0, 1)
    return ((far - near) * divs + near).permute(1, 2, 0)

def sampleStratified(near, far, count):
    device = near.device

    n = float(count)
    divs = torch.linspace(0.0, 1.0, count, device=device).expand(*near.shape, count).permute(2, 0, 1)
    # TODO: double-check ranges
    rand = torch.rand(count, *near.shape, device=device) / (n - 1)
    return ((far - near) * (divs + rand) * (n - 1) / n + near).permute(1, 2, 0)

# TODO: weighted sampling based on ray length
# Stack all rays & sample uniformly, then unstack (like trimesh area-weighted scheme)
