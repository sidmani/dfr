import numpy as np
import torch

# compute px^2 rays emitted by a camera looking from given angle
def camera_rays(phi, theta, px, fov):
    # D: the radial distance of the camera from the origin
    D = 1.0 / np.sin(fov / 2.0)

    # x: the cartesian vector position of the camera
    x = D * np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # enumerate all camera ray angles
    phis = np.tile(np.linspace(-fov / 2.0, fov / 2.0, num=px), (px, 1))
    thetas = np.transpose(phis)

    # compute the central angle between each ray and the camera-origin line
    # note that the cos(arccos(.)) has been canceled
    # center is the right-triangle distance
    # XXX: looks ok, but check again
    center = D * (np.cos(phis) + np.cos(thetas) - 1)

    # the quadratic formula radical term
    # mask very small values (tangent) and negative values (outside sphere)
    # NOTE: 1e-10 should be about 2x epsilon value used in ray integral step
    radicand = np.ma.masked_less((center ** 2 - D ** 2 + 1.0), 1e-10)
    difference = np.ma.sqrt(radicand)

    # two roots
    s_1 = center - difference
    s_2 = center + difference

    # the direction vectors in the frustum
    # cache some values
    tt = theta + thetas
    pp = phi + phis
    cos_pp = np.cos(pp)

    # compute the cartesian vectors and transpose to group points
    rays = -np.transpose(np.stack([
        cos_pp * np.sin(tt),
        np.sin(pp),
        cos_pp * np.cos(tt),
    ]), (2, 1, 0))

    return (x, rays, s_1, s_2)

# sampling schemes
def uniform_sample(s_1, s_2, count):
    divs = np.tile(np.linspace(0.0, 1.0, count, endpoint=False), (s_1.shape[0], 1))
    return (divs.T * (s_2 - s_1) + s_1).T

def random_sample(s_1, s_2, count):
    rands = np.random.rand(s_1.shape[0], count)
    return (rands.T * (s_2 - s_1) + s_1).T

def stratified_random_sample(s_1, s_2, count):
    divs = np.tile(np.linspace(0.0, 1.0, count, endpoint=False), (s_1.shape[0], 1))
    rands = np.random.rand(divs.shape) / float(count)
    return ((divs + rands).T * (s_2 - s_1) + s_1).T

# def raycast(model_cb, phi, theta, px=64, fov=np.pi / 3.0, sampler=uniform_sample):
    # x, vecs, s_1, s_2 = camera_rays(phi, theta, px, fov)
