import numpy as np
import torch
from dfr.raycast.shader import searchRays

# signed-distance function for the unit sphere
def MockSDF(data):
    pts = data[:, :3]
    return torch.norm(pts, dim=1) - 1.0

def test_searchRays_1_ray():
    batch = 2
    latents = torch.zeros(batch, 256)
    targets = torch.tensor([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -0.9],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 2.0],
        ]).repeat(batch, 1, 1).unsqueeze(1)
    assert targets.shape == (2, 1, 5, 3)
    out = searchRays(latents, targets, MockSDF, 10e-8)

    assert out.shape == (batch, 1, 3)
    assert torch.equal(out[0, 0, :], torch.tensor([0.0, 0.0, -0.9]))

def test_searchRays_many_rays():
    batch = 2
    latents = torch.zeros(batch, 256)
    targets = torch.tensor([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -0.9],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 2.0],
        ]).repeat(batch, 7, 1, 1)
    assert targets.shape == (2, 7, 5, 3)
    out = searchRays(latents, targets, MockSDF, 10e-8)

    assert out.shape == (batch, 7, 3)
    assert torch.equal(out[0, 0, :], torch.tensor([0.0, 0.0, -0.9]))

# def test_rayIntegral():
#     batch = 7
#     sampleCount = 20

#     (cameraD,
#      phiSpace,
#      thetaSpace,
#      segmentNear,
#      segmentFar) = buildFrustum(2 * np.pi / 3, 4, device=None)

#     phis = torch.linspace(-1.0, 1.0, batch)
#     thetas = torch.linspace(-1.0, 1.0, batch)

#     rays = enumerateRays(phis, thetas, phiSpace, thetaSpace)
#     samples = sampleUniform(segmentNear, segmentFar, sampleCount, device=None).unsqueeze(0)
#     cameraLoc = sphereToRect(phis, thetas, cameraD)

#     hitMask = (segmentFar - segmentNear) > 1e-10

#     print(rays[:, hitMask].shape)
#     print(samples[:, hitMask].shape)
#     scaledRays = scaleRays(rays[:, hitMask], samples[:, hitMask], cameraLoc)
#     latents = torch.zeros(batch, 256)

#     critPoints = searchRays(latents, scaledRays, MockSDF, 1e-10)
#     print(samples[:, hitMask].shape)
#     assert critPoints.shape == (batch, samples[:, hitMask].shape[0], 3)

#     obj0 = critPoints[0]
#     for i in range(4):
#         print(torch.norm(obj0[i]).item())

#     assert False
