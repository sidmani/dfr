import numpy as np
import torch
from dfr.raycast.frustum import Frustum, sphereToRect

def test_sphereToRect_zAxis():
    v = sphereToRect(torch.zeros(1), torch.zeros(1), 1.0)
    assert torch.allclose(v, torch.tensor([0.0, 0.0, 1.0]), atol=5e-7)

def test_sphereToRect_yAxis():
    v = sphereToRect(torch.tensor([np.pi / 2.0]), torch.zeros(1), 1.0)
    assert torch.allclose(v, torch.tensor([0.0, 1.0, 0.0]), atol=5e-7)

def test_sphereToRect_xAxis():
    v = sphereToRect(torch.zeros(1), torch.tensor([np.pi / 2.0]), 1.0)
    assert torch.allclose(v, torch.tensor([1.0, 0.0, 0.0]), atol=5e-7)

def test_sphereToRect_xAxis():
    v = sphereToRect(torch.zeros(1), torch.tensor([np.pi / 2.0]), 1.0)
    assert torch.allclose(v, torch.tensor([1.0, 0.0, 0.0]), atol=5e-7)

def test_buildFrustum_cameraD():
    cameraD = Frustum(2*np.pi/3, 4, device=None).cameraD
    assert cameraD > 1.0
    assert type(cameraD) == np.float64

def test_buildFrustum_viewField():
    f = Frustum(np.pi/3, 4, device=None)
    assert f.viewField.shape == (1, 4 * 4, 3, 1)
    vf = f.viewField.view(4, 4, 3)

    # check that these are repeated in the correct directions
    assert torch.equal(vf[:, 0, 1], vf[:, 3, 1])
    assert torch.equal(vf[0, :, 0], vf[3, :, 0])

def test_buildFrustum_quadrants():
    f = Frustum(np.pi/3, 4, device=None)

    # check quadrants are oriented correctly
    vf = f.viewField.view(4, 4, 3)
    assert vf[0, 0, 0] < 0
    assert vf[0, 0, 1] > 0

    assert vf[0, 3, 0] > 0
    assert vf[0, 3, 1] > 0

    assert vf[3, 0, 0] < 0
    assert vf[3, 0, 1] < 0

    assert vf[3, 3, 0] > 0
    assert vf[3, 3, 1] < 0

def test_buildFrustum_segment():
    f = Frustum(np.pi/2, 12, device=None)
    for i in range(12):
        for j in range(12):
            assert f.near[i, j] <= f.far[i, j]
            assert f.near[i, j] > 0.0
            # check that segment is shorter than diameter of sphere, with float err
            assert f.far[i, j] - f.near[i, j] < 2.0 + 10e-10
