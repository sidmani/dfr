import torch
from torch.cuda.amp import autocast
import numpy as np
from ..flags import Flags

def rotateAxes(phis, thetas):
    cos_theta = torch.cos(thetas).unsqueeze(1)
    sin_theta = torch.sin(thetas).unsqueeze(1)
    cos_phi = torch.cos(phis).unsqueeze(1)
    sin_phi = torch.sin(phis).unsqueeze(1)

    # composition of 2 transforms: rotate theta first, then phi
    # TODO: reverse rotation order
    return torch.cat([
        torch.cat([cos_theta, -sin_theta * sin_phi, cos_phi * sin_theta], dim=1).unsqueeze(2),
        torch.cat([torch.zeros_like(cos_phi), cos_phi, sin_phi], dim=1).unsqueeze(2),
        torch.cat([-sin_theta, -sin_phi * cos_theta, cos_phi * cos_theta], dim=1).unsqueeze(2),
    ], dim=2)

def makeRays(axes, px, D, fov, dtype):
    edgeLength = (D - 1) * np.tan(fov / 2)

    # offsets the edges so a 2n x 2n grid is evenly spaced within an n x n grid
    # for example, raycasting at 32x32 avg pooled to 16x16 should look very similar to raycasting at 16x16
    edge = edgeLength * (1. - 1. / px)

    xSpace = torch.linspace(-edge, edge, steps=px, dtype=dtype, device=axes.device).repeat(px, 1)[None, :, :, None]
    ySpace = -xSpace.transpose(1, 2)
    x = axes[:, 0][:, None, None, :]
    y = axes[:, 1][:, None, None, :]
    z = axes[:, 2][:, None, None, :]

    plane = xSpace * x + ySpace * y + z
    rays = plane - z * D
    norm = rays.reshape(-1, 3).norm(dim=1).view(-1, px, px, 1)
    return rays / norm

def computePlanes(rays, axes, cameraD, size):
    z = axes[:, 2][:, None, None, :]
    center = cameraD * (-z.unsqueeze(3) @ rays.unsqueeze(4)).view(-1, size, size)
    delta = torch.sqrt(torch.clamp(center ** 2 - cameraD ** 2 + 1, min=0.0))
    return center - delta, center + delta, delta > 1e-10

# nearest-neighbor upsampling
# can probably do a better job with bilinear interpolation
def upsample(t, scale):
    return t.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)

def multiscale(axes, scales, fov, latents, sdf, dtype, threshold):
    cameraD = 1.0 / np.sin(fov / 2.0)

    batch = axes.shape[0]
    terminal = torch.zeros(batch, 1, 1, 1, dtype=dtype, device=axes.device)
    rayMask = torch.ones_like(terminal, dtype=torch.bool).squeeze(3)
    origin = cameraD * axes[:, 2][:, None, None, :]
    latents = latents[:, None, None, :]
    size = 1

    smallestMask = None
    first = True
    for scale in scales:
        if not first:
            # TODO: move this into raymarch()
            # a geometric bound for whether a super-ray could have subrays that intersected the object
            k = distances * 2 * np.tan(fov / (2. * size))
            # this is suspicious because it uses a step size which is not actually used by sphere tracing
            # also the 0.5 at the end is empirical
            bound = torch.sqrt(2. * k ** 2. + (1.0 / 16.0) ** 2) * 0.5
            del k

            # subdivide the rays that pass close to the object
            rayMask[rayMask] = minValues <= bound
            del bound
            del minValues
        first = False
        size *= scale

        rays = makeRays(axes, size, cameraD, fov, dtype=dtype)
        near, far, sphereMask = computePlanes(rays, axes, cameraD, size)
        if smallestMask is None:
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
                                        threshold=threshold,
                                        dtype=dtype)

        terminal[rayMask] = distances.unsqueeze(1)

    points = origin + terminal * rays

    # use the lowest-resolution mask, because the high res mask includes unsampled rays
    sphereMask = upsample(smallestMask, size // scales[0]).expand(batch, -1, -1)
    return points, expandedLatents, sphereMask

def sphereTrace(rays, origin, planes, latents, sdf, threshold, dtype, steps=16):
    minValues = 5.0 * torch.ones(rays.shape[0], device=rays.device, dtype=dtype)
    mask = torch.ones_like(minValues, dtype=torch.bool)
    distance = planes[0].clone() + 2.0 / steps
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
            values = sdf(targets, latents, mask, geomOnly=True).squeeze(1).type(dtype)

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
