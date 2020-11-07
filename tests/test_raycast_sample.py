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

def test_scaleRays():
    batch = 7
    sampleCount = 5

    (cameraD,
     phiSpace,
     thetaSpace,
     segmentNear,
     segmentFar) = buildFrustum(2 * np.pi / 3, 4, device=None)

    phis = torch.rand(batch)
    thetas = torch.rand(batch)

    rays = enumerateRays(phis, thetas, phiSpace, thetaSpace)
    samples = sampleUniform(segmentNear, segmentFar, sampleCount, device=None)
    cameraLoc = sphereToRect(phis, thetas, cameraD)

    hitMask = (segmentFar - segmentNear) > 1e-10

    scaledRays = scaleRays(rays[:, hitMask], samples[hitMask], cameraLoc)
    assert scaledRays.shape == (batch, rays[:, hitMask].shape[1], sampleCount, 3)

    # check that all the points on each ray are collinear
    for j in range(scaledRays.shape[1]):
        # have to subtract cameraLoc because scaling is around origin
        ray0 = scaledRays[0, j, 0] - cameraLoc[0]
        ray0_unit = ray0 / torch.norm(ray0)

        for i in range(sampleCount):
            ray_i = scaledRays[0, j, i] - cameraLoc[0]
            ray_i_unit = ray_i / torch.norm(ray_i)
            assert torch.allclose(ray0_unit, ray_i_unit)

