import pytest
import numpy as np
import torch
from dfr.raycast import buildFrustum

def test_buildFrustum_cameraD():
    cameraD = buildFrustum(2*np.pi/3, 4)[0]
    assert cameraD > 1.0
    assert type(cameraD) == np.float64

def test_buildFrustum_angleSpace():
    _, phiSpace, thetaSpace, _, _ = buildFrustum(np.pi, 4)
    assert phiSpace.shape == (4, 4)
    assert thetaSpace.shape == (4, 4)

    # check that these are repeated in the correct directions
    assert torch.equal(phiSpace[:, 0], phiSpace[:, 3])
    assert torch.equal(thetaSpace[0, :], thetaSpace[3, :])

    # check quadrants are oriented correctly
    assert phiSpace[0, 0] > 0
    assert thetaSpace[0, 0] < 0
    assert phiSpace[3, 3] < 0
    assert thetaSpace[3, 3] > 0

def test_buildFrustum_segment():
    _, _, _, near, far = buildFrustum(np.pi/2, 12)
    for i in range(12):
        for j in range(12):
            assert near[i, j] <= far[i, j]
            assert near[i, j] > 0.0
            # check that segment is shorter than diameter of sphere, with float err
            assert far[i, j] - near[i, j] < 2.0 + 10e-10
