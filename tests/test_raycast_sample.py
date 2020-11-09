import numpy as np
import torch
from dfr.raycast.frustum import buildFrustum, enumerateRays, sphereToRect
from dfr.raycast.sample import sampleRandom, sampleUniform, sampleStratified, scaleRays

def test_sampleUniform():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleUniform(near, far, 5, device=None)
    assert samples.shape == (4, 4, 5)
    for i in range(4):
        for j in range(4):
            assert torch.equal(samples[i, j], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

def test_sampleRandom():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleRandom(near, far, 10, device=None)
    assert samples.shape == (4, 4, 10)
    for k in range(4):
        for j in range(4):
            # check sorted increasing along each ray
            prev = -1
            for i in range(10):
                assert samples[j, k, i] > prev
                prev = samples[j, k, i]

def test_sampleStratified():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleStratified(near, far, 5, device=None)
    assert samples.shape == (4, 4, 5)
    # check that samples are in evenly-spaced divisions
    for k in range(4):
        for j in range(4):
            for i in range(5):
                x = samples[j, k, i]
                assert float(i * 0.8) < x < float((i + 1) * 0.8)

def test_scaleRays_simple():
    rays = torch.ones(2, 5, 3)
    samples = torch.linspace(0.0, 1.0, 7).repeat(2, 5, 1)
    assert samples.shape == (2, 5, 7)

    scaledRays = scaleRays(rays, samples, torch.ones(2, 3))
    obj1 = scaledRays[0]
    ray1 = obj1[0]
    assert torch.equal(ray1, torch.linspace(0.0, 1.0, 7).repeat(3, 1).transpose(0, 1) + 1.0)