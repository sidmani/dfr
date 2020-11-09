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

def test_generator_raycast():
    g = Generator(
            weightNorm=False,
            fov=2.0,
            px=8,
            sampleCount=16,
            latentSize=16,
            )

    phis = torch.tensor([0.0])
    thetas = torch.tensor([0.0])
    latents = torch.zeros(1, 16)
    out = g(latents, phis, thetas)
    obj1 = out[0]
    # edges are background
    assert torch.equal(obj1[0, :], torch.ones(8))
    assert torch.equal(obj1[7, :], torch.ones(8))
    assert torch.equal(obj1[:, 0], torch.ones(8))
    assert torch.equal(obj1[:, 7], torch.ones(8))

    # center should be dark
    assert obj1[3, 3] < 0.5
    assert obj1[3, 4] < 0.5
    assert obj1[4, 3] < 0.5
    assert obj1[4, 4] < 0.5

def test_generator_sample():
    g = Generator(
            weightNorm=False,
            fov=2.0,
            px=8,
            sampleCount=16,
            latentSize=16,
            )

    obj1 = g.sample(2)[0]
    # edges are background
    assert torch.equal(obj1[0, :], torch.ones(8))
    assert torch.equal(obj1[7, :], torch.ones(8))
    assert torch.equal(obj1[:, 0], torch.ones(8))
    assert torch.equal(obj1[:, 7], torch.ones(8))

    # center should be dark
    assert obj1[3, 3] < 0.5
    assert obj1[3, 4] < 0.5
    assert obj1[4, 3] < 0.5
    assert obj1[4, 4] < 0.5

