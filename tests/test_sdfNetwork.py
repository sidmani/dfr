import torch
from dfr.sdfNetwork import SDFNetwork

def test_sdfNetwork():
    latentSize = 15
    batch = 4
    sdf = SDFNetwork(weightNorm=False, latentSize=latentSize)

    latents = torch.ones(batch, latentSize)
    coords = torch.ones(batch, 3)
    inp = torch.cat([coords, latents], dim=1)
    out = sdf(inp)
    assert out.shape == (batch, 1)
