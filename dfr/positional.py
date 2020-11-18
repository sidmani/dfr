import torch
import numpy as np

def positional(x, L=6):
    device = x.device

    basis = 2.0 ** torch.linspace(0, L - 1, L, device=device) * np.pi
    arg = torch.flatten(x.view(-1, 1, 3) * basis.view(1, L, 1), 1, 2)
    sin, cos = torch.sin(arg), torch.cos(arg)
    return torch.cat([sin, cos], dim=1)
