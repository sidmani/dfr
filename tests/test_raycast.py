import pytest
import numpy as np
import torch
from dfr.raycast import buildFrustum, enumerateRays, sampleRandom, sampleUniform, sampleStratified, sphereToRect

def test_sphereToRect_zAxis():
    v = sphereToRect(torch.zeros(1), torch.zeros(1), 1.0)
    assert torch.allclose(v, torch.tensor([0.0, 0.0, 1.0]), atol=5e-7)

def test_sphereToRect_yAxis():
    v = sphereToRect(torch.zeros(1), torch.tensor([np.pi / 2.0]), 1.0)
    assert torch.allclose(v, torch.tensor([0.0, 1.0, 0.0]), atol=5e-7)

def test_sphereToRect_xAxis():
    v = sphereToRect(torch.tensor([np.pi / 2.0]), torch.zeros(1), 1.0)
    assert torch.allclose(v, torch.tensor([1.0, 0.0, 0.0]), atol=5e-7)

def test_buildFrustum_cameraD():
    cameraD = buildFrustum(2*np.pi/3, 4)[0]
    assert cameraD > 1.0
    assert type(cameraD) == np.float64

def test_buildFrustum_angleSpace():
    _, phiSpace, thetaSpace, _, _ = buildFrustum(np.pi/3, 4)
    assert phiSpace.shape == (4, 4)
    assert thetaSpace.shape == (4, 4)

    # check that these are repeated in the correct directions
    assert torch.equal(phiSpace[:, 0], phiSpace[:, 3])
    assert torch.equal(thetaSpace[0, :], thetaSpace[3, :])

def test_buildFrustum_quadrants():
    _, phiSpace, thetaSpace, _, _ = buildFrustum(np.pi/3, 4)

    # check quadrants are oriented correctly
    assert 0 < thetaSpace[3, 3] < np.pi
    assert np.pi < thetaSpace[0, 0] < np.pi * 2

    assert 0 < phiSpace[0, 0] < np.pi / 2
    assert -np.pi / 2 < phiSpace[3, 3] < 0

def test_buildFrustum_segment():
    _, _, _, near, far = buildFrustum(np.pi/2, 12)
    for i in range(12):
        for j in range(12):
            assert near[i, j] <= far[i, j]
            assert near[i, j] > 0.0
            # check that segment is shorter than diameter of sphere, with float err
            assert far[i, j] - near[i, j] < 2.0 + 10e-10

def test_enumerateRays_shape():
    batch_size = 5
    px = 4

    phis = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thetas = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    _, phiSpace, thetaSpace, _, _ = buildFrustum(2*np.pi/3, px)
    rays = enumerateRays(phis, thetas, phiSpace, thetaSpace)
    assert rays.shape == (batch_size, px, px, 3)

def test_enumerateRays_zMatch():
    batch_size = 5
    px = 4

    phis = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thetas = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    _, phiSpace, thetaSpace, _, _ = buildFrustum(2*np.pi/3, px)
    rays = enumerateRays(phis, thetas, phiSpace, thetaSpace)

    first = rays[0]
    assert first.shape == (px, px, 3)

    # . * * .
    # * . . *
    # * . . *
    # . * * .
    # starred rays should have same z value
    assert torch.allclose(first[0, 1, 2], first[0, 2, 2])
    assert torch.allclose(first[0, 1, 2], first[1, 0, 2])
    assert torch.allclose(first[0, 1, 2], first[1, 3, 2])
    assert torch.allclose(first[0, 1, 2], first[2, 0, 2])
    assert torch.allclose(first[0, 1, 2], first[2, 3, 2])
    assert torch.allclose(first[0, 1, 2], first[3, 1, 2])
    assert torch.allclose(first[0, 1, 2], first[3, 2, 2])

def test_enumerateRays_signs():
    batch_size = 5
    px = 4

    phis = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thetas = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    _, phiSpace, thetaSpace, _, _ = buildFrustum(2*np.pi/3, px)
    rays = enumerateRays(phis, thetas, phiSpace, thetaSpace)

    first = rays[0]
    # all z values are negative
    for i in range(4):
        for j in range(4):
            assert first[i, j, 2] < 0

    # y axis is positive for top half of image, negative for bottom half
    for i in range(2):
        for j in range(4):
            assert first[i, j, 1] > 0

    for i in range(2):
        for j in range(4):
            assert first[i+2, j, 1] < 0

    # x axis is negative for left half, positive for right half
    for i in range(4):
        for j in range(2):
            assert first[i, j, 0] < 0

    for i in range(4):
        for j in range(2):
            assert first[i, j+2, 0] > 0

def test_sampleUniform():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleUniform(near, far, 5, device=None)
    assert samples.shape == (4, 4, 5)
    assert torch.equal(samples[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

def test_sampleRandom():
    near = torch.zeros(4, 4)
    far = torch.ones(4, 4) * 4.0
    samples = sampleRandom(near, far, 5, device=None)
    assert samples.shape == (4, 4, 5)
    # check sorted increasing along each ray
    prev = -1
    for i in range(5):
        assert samples[0, 0, i] > prev
        prev = samples[0, 0, i]

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
