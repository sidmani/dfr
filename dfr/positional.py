import torch
import numpy as np

# positional encoding from NeRF
def positional(x, L=6):
    # [2^0 pi, 2^1 pi, ...]
    basis = (2.0 ** torch.linspace(0, L - 1, L, device=x.device)) * np.pi
    # multiply each point by each basis element
    arg = torch.flatten(x.view(-1, 1, 3) * basis.view(1, L, 1), 1, 2)
    sin, cos = torch.sin(arg), torch.cos(arg)
    return torch.cat([sin, cos], dim=1)
