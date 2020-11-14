import numpy as np
import torch
from dfr.raycast.sample import sampleRandom, sampleUniform, sampleStratified

def test_sampleUniform():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleUniform(near, far, 5)
    assert samples.shape == (4, 4, 5)
    for i in range(4):
        for j in range(4):
            assert torch.equal(samples[i, j], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

def test_sampleRandom():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleRandom(near, far, 10)
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
    samples = sampleStratified(near, far, 5)
    assert samples.shape == (4, 4, 5)
    # check that samples are in evenly-spaced divisions
    for k in range(4):
        for j in range(4):
            for i in range(5):
                x = samples[j, k, i]
                assert float(i * 0.8) < x < float((i + 1) * 0.8)
