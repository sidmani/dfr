import torch
from dfr.sdfNetwork import SDFNetwork
from dfr.checkpoint import HParams

def test_sdfNetwork():
    latentSize = 256
    batch = 9
    ptsCount = 3
    sdf = SDFNetwork(HParams())

    latents = torch.ones(batch // ptsCount, latentSize)
    coords = torch.ones(batch, 3)
    out = sdf(coords, latents)
    assert out.shape == (batch, 1)
