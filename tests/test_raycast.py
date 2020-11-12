import torch
import numpy as np
from dfr.sdfNetwork import SDFNetwork
from dfr.checkpoint import HParams
from dfr.raycast.frustum import Frustum, enumerateRays, sphereToRect
from dfr.raycast.sample import sampleUniform, scaleRays
from dfr.raycast.shader import fastRayIntegral, shade
from dfr.generator import Generator

# signed-distance function for the half-unit sphere
class MockSDF:
    def __init__(self):
        self.hparams = HParams()

    def __call__(self, x, latents):
        return torch.norm(x, dim=1) - 0.5

# make sure a sphere shows up without the sphere masking
def test_raycast_sphere_manual():
    phis = torch.tensor([0.0])
    thetas = torch.tensor([0.0])
    latents = torch.zeros(1, 256)
    hp = HParams(imageSize=8)

    frustum = Frustum(2.0 * np.pi / 3.0, 8, device=None)
    sdf = MockSDF()

    # build a rotated frustum for each input angle
    rays = enumerateRays(phis, thetas, frustum.viewField, 8)

    # uniformly sample distances from the camera in the unit sphere
    # unsqueeze because we're using the same sample values for all objects
    samples = sampleUniform(
            frustum.near,
            frustum.far,
            hp.raySamples).unsqueeze(0)

    # compute the sampling points for each ray that intersects the unit sphere
    cameraLoc = sphereToRect(phis, thetas, frustum.cameraD)
    targets = scaleRays(
            rays.view(1, -1, 3),
            samples.reshape(1, -1, 32),
            cameraLoc)

    # compute intersections for rays
    values = fastRayIntegral(latents, targets, sdf, 10e-10).view(1, 8, 8)

    # shape [px, px, channels]
    out = shade(values)

    obj1 = out[0]
    # edges are background
    assert torch.allclose(obj1[0, :], torch.ones(8))
    assert torch.allclose(obj1[7, :], torch.ones(8))
    assert torch.allclose(obj1[:, 0], torch.ones(8))
    assert torch.allclose(obj1[:, 7], torch.ones(8))

    # center should be dark
    assert obj1[3, 3] < 0.5
    assert obj1[3, 4] < 0.5
    assert obj1[4, 3] < 0.5
    assert obj1[4, 4] < 0.5

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
