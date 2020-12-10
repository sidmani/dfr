import torch
import numpy as np

def createBasis(L, sigma, device):
    # create an Lx3 matrix
    # by the function of random variables formula, multiplying by
    # the SD makes the result distributed with that SD
    return 2. * np.pi * torch.randn(L, 3, device=device) * sigma

# positional encoding from NeRF
def positional(x, basis, inplace=False):
    arg = x @ basis.T
    sin = torch.sin(arg)

    if inplace:
        cos = torch.cos_(arg)
    else:
        cos = torch.cos(arg)

    return sin, cos
