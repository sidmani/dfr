import torch
from .frustum import enumerateRays, sphereToRect
from .shader import fastRayIntegral, shade
from .sample import sampleUniform, scaleRays

def raycast(sdf, latents, phis, thetas, frustum, sampleCount, device):
    # build a rotated frustum for each input angle
    rays = enumerateRays(phis, thetas, frustum.phiSpace, frustum.thetaSpace)

    # uniformly sample distances from the camera in the unit sphere
    # unsqueeze because we're using the same sample values for all objects
    samples = sampleUniform(
            frustum.near,
            frustum.far,
            sampleCount,
            device).unsqueeze(0)

    # compute the sampling points for each ray that intersects the unit sphere
    cameraLoc = sphereToRect(phis, thetas, frustum.cameraD)
    targets = scaleRays(
            rays[:, frustum.mask],
            samples[:, frustum.mask],
            cameraLoc)

    # compute intersections for rays
    values = fastRayIntegral(latents, targets, sdf, 10e-10)

    # shape [px, px, channels]
    result = torch.ones(rays.shape[:3])
    result[:, frustum.mask] = shade(values)
    return result
