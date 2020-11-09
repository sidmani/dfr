import torch
import numpy as np
from dfr.raycast import raycast
from dfr.raycast.frustum import buildFrustum
from dfr.sdfNetwork import SDFNetwork

# signed-distance function for the half-unit sphere
def MockSDF(pts, latents):
    return torch.norm(pts, dim=1) - 0.5

def test_raycast_sphere():
    frustum = buildFrustum(2.0 * np.pi / 3.0, 8, device=None)
    phis = torch.tensor([0.0])
    thetas = torch.tensor([0.0])
    latents = torch.zeros(1, 256)

    out = raycast(MockSDF, latents, phis, thetas, frustum, 10, device=None)
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
    frustum = buildFrustum(2.0 * np.pi / 3.0, 8, device=None)
    phis = torch.tensor([0.0])
    thetas = torch.tensor([0.0])
    latents = torch.zeros(1, 256)

    out = raycast(SDFNetwork(weightNorm=False, latentSize=256), latents, phis, thetas, frustum, 10, device=None)
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
