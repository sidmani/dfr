import numpy as np
import torch
from dfr.raycast.ray import rotateFrustum
from dfr.raycast.frustum import Frustum

def test_rotateFrustum_shape():
    batch_size = 5
    px = 4

    phis = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thetas = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    f = Frustum(2*np.pi/3, px, device=None)
    rays, cameraLoc = rotateFrustum(phis, thetas, f)
    assert rays.shape == (batch_size, px, px, 3)

def test_rotateFrustum_zMatch():
    batch_size = 5
    px = 4

    phis = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thetas = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    f = Frustum(2*np.pi/3, px, device=None)
    rays, cameraLoc = rotateFrustum(phis, thetas, f)

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

def test_rotateFrustum_signs():
    px = 4

    phis = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thetas = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    f = Frustum(2*np.pi/3, px, device=None)
    rays, cameraLoc = rotateFrustum(phis, thetas, f)

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

def test_rotateFrustum_signs_theta_pi():
    px = 4

    phis = torch.tensor([0.0])
    thetas = torch.tensor([np.pi])
    f = Frustum(2*np.pi/3, px, device=None)
    rays, cameraLoc = rotateFrustum(phis, thetas, f)

    first = rays[0]
    # all z values are positive
    for i in range(4):
        for j in range(4):
            assert first[i, j, 2] > 0

    # y axis is positive for top half of image, negative for bottom half
    for i in range(2):
        for j in range(4):
            assert first[i, j, 1] > 0

    for i in range(2):
        for j in range(4):
            assert first[i+2, j, 1] < 0

    # x axis is positive for left half, negative for right half
    for i in range(4):
        for j in range(2):
            assert first[i, j, 0] > 0

    for i in range(4):
        for j in range(2):
            assert first[i, j+2, 0] < 0

def test_rotateFrustum_signs_theta_nonzero():
    px = 3

    phis = torch.tensor([0.0])
    thetas = torch.tensor([np.pi / 6])
    f = Frustum(np.pi/3, px, device=None)
    rays, cameraLoc = rotateFrustum(phis, thetas, f)

    first = rays[0]
    # all z values are negative
    for i in range(3):
        for j in range(3):
            assert first[i, j, 2] < 0

    # right center pixel should be negative z axis
    assert torch.allclose(first[1, 2], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)

def test_rotateFrustum_signs_phi_nonzero():
    px = 3
    phis = torch.tensor([np.pi/6])
    thetas = torch.tensor([0.0])
    f = Frustum(np.pi/3, px, device=None)
    rays, cameraLoc = rotateFrustum(phis, thetas, f)

    first = rays[0]
    # all z values are negative
    for i in range(3):
        for j in range(3):
            assert first[i, j, 2] < 0

    # the top center pixel's ray should be the negative z axis
    assert torch.allclose(first[0, 1], torch.tensor([0.0, 0.0, -1.0]), atol=1e-7)
