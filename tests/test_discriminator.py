import torch
from dfr.discriminator import Discriminator

def test_discriminator():
    d = Discriminator()
    t = torch.ones(3, 64, 64)
    out = d(t)
    assert out.shape == (3,)

    for i in range(3):
        assert 0.0 < out[i] < 1.0
