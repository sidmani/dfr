import torch
from torch.cuda.amp import autocast
import numpy as np
from ..flags import Flags
from .geometry import rayGrid, computePlanes
from collections import namedtuple

SampleData = namedtuple('SampleData', ['points', 'latents', 'mask'])

# nearest-neighbor upsampling
# TODO: can probably do a better job with bilinear interpolation
def upsample(t, scale):
    return t.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)

# iteratively raycast at increasing resolution
def multiscale(axes, scales, latents, sdf, threshold, fov=25 * (np.pi / 180)):
    cameraD = 1.0 / np.tan(fov / 2.0)

    batch = axes.shape[0]
    terminal = torch.zeros(batch, 1, 1, 1, device=axes.device)
    rayMask = torch.ones_like(terminal, dtype=torch.bool).squeeze(3)
    origin = cameraD * axes[:, 2][:, None, None, :]
    latents = latents[:, None, None, :]
    size = 1

    for idx, scale in enumerate(scales):
        if idx > 0:
            # TODO: move this into raymarch()
            # a geometric bound for whether a super-ray could have subrays that intersected the object
            k = distances * 2 * np.tan(fov / (2. * size))
            # this is suspicious because it uses a step size which is not actually used by sphere tracing
            # also the 0.5 at the end is empirical
            bound = torch.sqrt(2. * k ** 2. + (1.0 / 16.0) ** 2) * 0.5

            # subdivide the rays that pass close to the object
            rayMask[rayMask] = minValues <= bound

        size *= scale
        rays = rayGrid(axes, size, cameraD)
        near, far, sphereMask = computePlanes(rays, axes, cameraD, size)
        if idx == 0:
            smallestMask = sphereMask

        near = near.expand(batch, -1, -1)
        far = far.expand(batch, -1, -1)
        terminal = upsample(terminal, scale)
        rayMask = upsample(rayMask, scale)
        expandedLatents = latents.expand(-1, size, size, -1)
        expandedOrigin = origin.expand(-1, size, size, -1)

        # terminate rays that don't intersect the unit sphere
        rayMask = torch.logical_and(rayMask, sphereMask)
        planes = torch.stack([near[rayMask], far[rayMask]])

        minValues, distances = sphereTrace(rays[rayMask],
                                        expandedOrigin[rayMask],
                                        planes,
                                        expandedLatents[rayMask],
                                        sdf,
                                        threshold=threshold)

        terminal[rayMask] = distances.unsqueeze(1)

    # use the lowest-resolution mask, because the high res mask includes unsampled rays
    sphereMask = upsample(smallestMask, size // scales[0]).expand(batch, -1, -1)
    points = (origin + terminal * rays)[sphereMask]
    points.requires_grad = True
    return SampleData(points, expandedLatents, sphereMask)

# batched sphere tracing with culling of terminated rays
def sphereTrace(rays, origin, planes, latents, sdf, threshold, steps=16):
    minValues = 5.0 * torch.ones(rays.shape[0], device=rays.device)
    mask = torch.ones_like(minValues, dtype=torch.bool)
    distance = planes[0] + 2.0 / steps
    minDistances = distance.clone()

    # start from 1, because the initial critical points are the 0-idx
    for i in range(1, steps):
        # march along each ray
        # compute the next target point
        targets = (origin + rays * distance.unsqueeze(1))[mask]

        # stop if all rays have terminated
        if targets.shape[0] == 0:
            break

        # evaluate the targets
        with autocast(enabled=Flags.AMP):
            values = sdf(targets, latents, mask, geomOnly=True).squeeze(1).type(torch.float)

        del targets

        # TODO: this is the bottleneck
        # Boolean indexing produces a variably-sized result, which causes an
        # expensive CPU-GPU sync. But all of these masks work together to limit the
        # number of SDF queries, which is also very expensive.
        minMask = values < minValues[mask]
        updateMask = torch.zeros_like(mask)
        updateMask[mask] = minMask

        floatMask = updateMask.float()
        # update min and argmin
        minDistances = (1 - floatMask) * minDistances + floatMask * distance
        minValues[updateMask] = values[minMask]

        # terminate all rays that intersect the surface (negated)
        intersectMask = values > threshold
        distance[mask] += values

        # terminate rays that exit the unit sphere on the next step (again negated)
        exitMask = distance < planes[1] * mask.float()

        mask[mask] = intersectMask
        mask = torch.logical_and(mask, exitMask)

    return minValues, minDistances
