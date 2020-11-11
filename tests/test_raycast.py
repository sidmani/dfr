import torch
import numpy as np
from dfr.sdfNetwork import SDFNetwork
from dfr.checkpoint import HParams
from dfr.raycast.frustum import Frustum
from dfr.generator import Generator

# signed-distance function for the half-unit sphere
class MockSDF:
    def __init__(self):
        self.hparams = HParams()

    def __call__(self, x, latents):
        return torch.norm(x, dim=1) - 0.5

def test_raycast_sphere():
    phis = torch.tensor([0.0])
    thetas = torch.tensor([0.0])
    latents = torch.zeros(1, 256)
    hp = HParams(imageSize=8)

    frustum = Frustum(2.0 * np.pi / 3.0, 8, device=None)
    gen = Generator(MockSDF(), frustum, hp)
    out = gen.raycast(latents, phis, thetas)

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

def test_raycast_realSDF():
    phis = torch.tensor([0.0])
    thetas = torch.tensor([0.0])
    latents = torch.zeros(1, 256)
    hp = HParams()

    frustum = Frustum(2.0 * np.pi / 3.0, hp.imageSize, device=None)
    gen = Generator(MockSDF(), frustum, hp)
    out = gen.raycast(latents, phis, thetas)

    obj1 = out[0]
    # edges are background
    assert torch.allclose(obj1[0, :], torch.ones(hp.imageSize))
    assert torch.allclose(obj1[hp.imageSize-1, :], torch.ones(hp.imageSize))
    assert torch.allclose(obj1[:, 0], torch.ones(hp.imageSize))
    assert torch.allclose(obj1[:, hp.imageSize-1], torch.ones(hp.imageSize))

    # center should be dark
    for j in range(-1, 1):
        for i in range(-1, 1):
            assert obj1[hp.imageSize // 2 + i, hp.imageSize // 2 + j] < 0.5
