import torch
from dfr.raycast.ray import distributeSamples

def test_distributeSamples_simple():
    rays = torch.ones(2, 5, 3)
    samples = torch.linspace(0.0, 1.0, 7).repeat(2, 5, 1)
    assert samples.shape == (2, 5, 7)

    scaledRays = distributeSamples(rays, samples, torch.ones(2, 3))
    obj1 = scaledRays[0]
    ray1 = obj1[0]
    assert torch.equal(ray1, torch.linspace(0.0, 1.0, 7).repeat(3, 1).transpose(0, 1) + 1.0)
