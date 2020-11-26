import torch
from torch.nn.functional import avg_pool2d, interpolate
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

def intersection(rays, frustum, cameraLoc, latents, sdf, coarseSteps=32, fineSteps=32, kernel=4):
    imageSize = (frustum.imageSize, frustum.imageSize)
    smallSize = (frustum.imageSize // kernel, frustum.imageSize // kernel)
    batch = rays.shape[0]
    coarseStepSize = 2.0 / float(coarseSteps)

    # first pass pools rays and uses a large step size to figure out approx. where the object is
    pooledRays = avg_pool2d(rays.permute(0, 3, 1, 2), kernel_size=kernel).permute(0, 2, 3, 1)
    pooledRays = pooledRays.view(batch, -1, 3)
    pooledNear = avg_pool2d(frustum.near[None, None, ...], kernel_size=kernel).view(1, -1, 1)

    origin = cameraLoc.view(-1, 1, 3) + pooledRays * pooledNear
    expandedLatents = latents.unsqueeze(1).expand(-1, pooledRays.shape[1], -1)
    coarsePoints, minValues, distances = raymarch(pooledRays,
                                       origin,
                                       expandedLatents,
                                       sdf,
                                       steps=coarseSteps,
                                       stepSize=coarseStepSize)

    minValues = minValues.view(-1, 1, *smallSize)
    coarsePoints = coarsePoints.view(-1, *smallSize, 3).permute(0, 3, 1, 2)
    distances = distances.view(-1, 1, *smallSize)
    # upsample the results
    minValues = interpolate(minValues, size=imageSize).squeeze(1)
    coarsePoints = interpolate(coarsePoints, size=imageSize).permute(0, 2, 3, 1)
    distances = interpolate(distances, size=imageSize).squeeze(1)

    # figure out which rays need to be computed
    # if the min distance is less than half the step size,
    k = float(kernel) / float(frustum.imageSize) * frustum.fov * (distances + frustum.near)
    bound = 0.5 * torch.sqrt(2. * k ** 2. + coarseStepSize ** 2)
    hitMask = minValues <= bound

    origin = cameraLoc.view(-1, 1, 1, 3) + rays * frustum.near[None, :, :, None]
    expandedLatents = latents[:, None, None, :].expand(-1, *rays.shape[1:3], -1)

    critPoints, minValues, _ = raymarch(rays[hitMask].unsqueeze(0),
                             origin[hitMask].unsqueeze(0),
                             expandedLatents[hitMask].unsqueeze(0),
                             sdf,
                             steps=fineSteps,
                             stepSize=2.0 / float(fineSteps))

    coarsePoints[hitMask] = critPoints + rays[hitMask].unsqueeze(0) * minValues.unsqueeze(2)
    missedRays = rays[~hitMask]
    return coarsePoints

def raymarch(rays, origin, latents, sdf, steps, stepSize):
    device = rays.device
    mask = torch.ones(*rays.shape[:2], dtype=torch.bool, device=device)
    origin = origin + rays * torch.rand(1, rays.shape[1], 1, device=device) * stepSize
    critPoints = origin + rays * stepSize

    distances = 5.0 * torch.ones_like(mask, dtype=torch.float)
    minValues = distances.clone()

    # start from 1, because the initial critical points are the 0-idx
    for i in range(1, steps):
        # march along each ray
        dist = float(i) * stepSize
        distances[mask] = dist
        # compute the next target points
        targets = (origin + rays * dist)[mask]

        # stop if all rays have terminated
        if targets.shape[0] == 0:
            break

        # evaluate the targets
        values = sdf(targets, latents[mask], geomOnly=True).squeeze(1)
        minMask = values < minValues[mask]
        updateMask = torch.zeros_like(mask)
        updateMask[mask] = minMask

        # update min and argmin
        minValues[updateMask] = values[minMask]
        critPoints[updateMask] = targets[minMask]

        # terminate all rays that intersect the surface
        # note that this is non-increasing; rays can't come back to life
        mask[mask] = values > 0.0

    return critPoints, minValues, distances
