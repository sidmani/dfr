import torch
from dfr.sdfNetwork import SDFNetwork

def test_sdfNetwork():
    latentSize = 15
    batch = 9
    ptsCount = 3
    sdf = SDFNetwork(weightNorm=False, latentSize=latentSize)

    latents = torch.ones(batch // ptsCount, latentSize)
    coords = torch.ones(batch, 3)
    out = sdf(coords, latents)
    assert out.shape == (batch, 1)
