import torch
from .frustum import sphereToRect

# list the rays given a base frustum and a view angle
def rotateFrustum(phis, thetas, frustum, jitter):
    # cache some stuff that's reused
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    cos_phi = torch.cos(phis)
    sin_phi = torch.sin(phis)
    cos_phi_sin_theta = cos_phi * sin_theta
    cos_phi_cos_theta = cos_phi * cos_theta

    # create a 3D rotation matrix for each phi, theta pair
    rotation = torch.stack([
        torch.stack([cos_theta, -sin_theta * sin_phi, cos_phi_sin_theta]),
        torch.stack([torch.zeros_like(phis), cos_phi, sin_phi]),
        torch.stack([-sin_theta, -sin_phi * cos_theta, cos_phi_cos_theta]),
    ]).permute(2, 0, 1).unsqueeze(1)

    # have the cos and sin needed to compute camera location, so let's do it
    cameraLoc = frustum.cameraD * torch.stack([
        cos_phi_sin_theta,
        sin_phi,
        cos_phi_cos_theta,
    ], dim=-1)

    viewField = frustum.jitteredViewField() if jitter else frustum.viewField

    # rotate the frustum and view as [batch, px, px, 3] i.e. a ray for each pixel
    rays = torch.matmul(rotation, viewField).view(-1, frustum.imageSize, frustum.imageSize, 3)
    return rays, cameraLoc

# march along the rays to find the surface
# can't sphere trace, because early in training the SDF is ill-formed
# uses a lot of in-place operations and probably won't work with autograd
# TODO: can uses a ray step decay scheme to decrease samples further
def iterativeIntersection(rays, frustum, cameraLoc, latents, sdf, steps=32):
    device = rays.device
    stepSize = 2.0 / float(steps)
    cameraLoc = cameraLoc.view(-1, 1, 3)
    sphereRays = rays[:, frustum.mask]

    far = frustum.far[frustum.mask]
    # jitter the starting points by stepSize to avoid grid artifacts
    near = frustum.near[frustum.mask] + torch.rand(*far.shape, device=device) * stepSize
    mask = torch.ones(*sphereRays.shape[:2], dtype=torch.bool, device=device)
    expandedLatents = latents.unsqueeze(1).expand(-1, near.shape[0], -1)

    # initial critical points are on surface of sphere
    critPoints = cameraLoc + sphereRays * near.view(1, -1, 1)

    # minimum values are in [-1, 1] so start at 5 to guarantee decrease
    minValues = 5.0 * torch.ones(*sphereRays.shape[:2], device=device)

    # start from 1, because the initial critical points are the 0-idx
    for i in range(1, steps):
        # march along each ray 2/steps at a time
        # ray length <= 2.0 (diameter of unit sphere)
        distanceInSphere = near + float(i) * stepSize
        # kill the rays that have marched out of the unit sphere
        mask[:, distanceInSphere >= far] = False
        # compute the next target points
        targets = (cameraLoc + sphereRays * distanceInSphere.view(1, -1, 1))[mask]

        # stop if all rays have terminated
        if targets.shape[0] == 0:
            break

        # evaluate the targets
        values = sdf(targets, expandedLatents[mask], geomOnly=True).squeeze(1)

        # construct composite mask to update minimums
        minMask = values < minValues[mask]
        updateMask = torch.zeros_like(mask)
        updateMask[mask] = minMask

        # update min and argmin
        minValues[updateMask] = values[minMask]
        critPoints[updateMask] = targets[minMask]

        # terminate all rays that intersect the surface
        # note that this is non-increasing; rays can't come back to life
        mask[mask] = values > 0.0

    # for the intersected rays, adjust to match surface more closely
    # doesn't require sdf pass and significantly improves accuracy
    hitMask = minValues <= 0.0
    critPoints[hitMask] = sphereRays[hitMask] * minValues[hitMask].unsqueeze(1) + critPoints[hitMask]
    # save and return the hit mask, because adjustment invalidates checking < 0 on the result
    return critPoints, hitMask
