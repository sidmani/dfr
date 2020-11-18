import numpy as np
import torch
from dfr.raycast.ray import findIntersection

# signed-distance function for the unit sphere
def MockSDF(pts, latents, geomOnly=False):
    return torch.norm(pts[:, :3], dim=1) - 1.0

def test_findIntersection_1_ray():
    batch = 2
    latents = torch.zeros(batch, 256)
    targets = torch.tensor([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -0.9],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 2.0],
        ]).repeat(batch, 1, 1).unsqueeze(1)
    assert targets.shape == (2, 1, 5, 3)
    out = findIntersection(latents, targets, MockSDF, 10e-8)

    assert out.shape == (batch, 1, 3)
    assert torch.equal(out[0, 0, :], torch.tensor([0.0, 0.0, -0.9]))

def test_findIntersection_many_rays():
    batch = 2
    latents = torch.zeros(batch, 256)
    targets = torch.tensor([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -0.9],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 2.0],
        ]).repeat(batch, 7, 1, 1)
    assert targets.shape == (2, 7, 5, 3)
    out = findIntersection(latents, targets, MockSDF, 10e-8)

    assert out.shape == (batch, 7, 3)
    assert torch.equal(out[0, 0, :], torch.tensor([0.0, 0.0, -0.9]))
