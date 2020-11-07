import numpy as np
import torch
from dfr.raycast.frustum import buildFrustum, enumerateRays, sphereToRect
from dfr.raycast.sample import sampleRandom, sampleUniform, sampleStratified, scaleRays
from dfr.raycast.shader import searchRays

def MockSDF(data):
    pts = data[:, :3]
    print(pts)
    return torch.norm(pts, dim=1) - 1.0

def test_rayIntegral():
    batch = 7
    sampleCount = 20

    (cameraD,
     phiSpace,
     thetaSpace,
     segmentNear,
     segmentFar) = buildFrustum(2 * np.pi / 3, 4, device=None)

    phis = torch.linspace(-1.0, 1.0, batch)
    thetas = torch.linspace(-1.0, 1.0, batch)

    rays = enumerateRays(phis, thetas, phiSpace, thetaSpace)
    samples = sampleUniform(segmentNear, segmentFar, sampleCount, device=None)
    cameraLoc = sphereToRect(phis, thetas, cameraD)

    hitMask = (segmentFar - segmentNear) > 1e-10

    scaledRays = scaleRays(rays[:, hitMask], samples[hitMask], cameraLoc)
    latents = torch.zeros(batch, 256)

    critPoints = searchRays(latents, scaledRays, MockSDF, 1e-10)
    assert critPoints.shape == (batch, samples[hitMask].shape[0], 3)

#     obj0 = critPoints[0]
#     for i in range(4):
#         print(torch.norm(obj0[i]).item())

#     assert False
