import torch

def normalizedZ(shape, device):
    z = torch.normal(0.0, 1.0, size=shape, device=device)
    zNorm = z.norm(dim=1).unsqueeze(1)
    return z / (zNorm + 1e-5)
