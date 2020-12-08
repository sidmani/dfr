import torch
import numpy as np

def createBasis(L, device):
    # [2^0 pi, 2^1 pi, ...]
    b = (2.0 ** torch.linspace(0, L - 1, L, device=device)) * np.pi
    return b.view(1, L, 1)

# positional encoding from NeRF
def positional(x, basis, inplace=False):
    arg = (x.view(-1, 1, 3) * basis).view(-1, 3 * basis.shape[1])
    sin = torch.sin(arg)
    # TODO: do the second function in-place
    if inplace:
        cos = torch.cos_(arg)
    else:
        cos = torch.cos(arg)
    return sin, cos
