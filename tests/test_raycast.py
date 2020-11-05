import pytest
import numpy as np
from dfr.raycast import camera_rays

def test_camera_rays_x():
    x, vecs, s_1, s_2 = camera_rays(0.0, 0.0, 5, np.pi / 3.0)
    assert np.allclose(x, np.array([0.0, 0.0, 2.0]))

def test_camera_rays_vecs():
    x, vecs, s_1, s_2 = camera_rays(0.0, 0.0, 5, np.pi / 3.0)
    assert np.allclose(vecs[2, 2], np.array([0.0, 0.0, -1.0]))

    # sanity checks on vectors
    assert vecs.shape == (5, 5, 3)

def test_camera_rays_y_sym():
    x, vecs, s_1, s_2 = camera_rays(0.0, 0.0, 5, np.pi / 3.0)

    # symmetry of y axis over xz plane
    tmp_1 = vecs[4]
    tmp_1[:, 1] = -tmp_1[:, 1]
    assert np.array_equal(tmp_1, vecs[0])

def test_camera_rays_bounds():
    x, vecs, s_1, s_2 = camera_rays(0.0, 0.0, 5, np.pi / 3.0)
    assert s_1.shape == (5, 5)
    assert s_2.shape == (5, 5)
    assert np.all(np.greater(s_2, s_1))
    assert np.isclose(s_1[2, 2], 1.0)
    assert np.isclose(s_2[2, 2], 3.0)
