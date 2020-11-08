import torch
from dfr.generator import Generator

def test_generator():
    g = Generator(
            weightNorm=False,
            fov=2.0,
            px=24,
            sampleCount=16,
            latentSize=16,
            )

    latents = torch.rand(5, 16)
    phis = torch.rand(5, 1)
    thetas = torch.rand(5, 1)

    out = g(latents, phis, thetas)
    assert out.shape == (5, 24, 24)
