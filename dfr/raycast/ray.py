import torch
from torch.cuda.amp import autocast
import numpy as np

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

def makeRays(axes, px, D, edge, dtype):
    # TODO: offset
    xSpace = torch.linspace(-edge, edge, steps=px, dtype=dtype, device=axes.device).repeat(px, 1)[None, :, :, None]
    ySpace = -xSpace.transpose(1, 2)
    x = axes[:, 0][:, None, None, :]
    y = axes[:, 1][:, None, None, :]
    z = axes[:, 2][:, None, None, :]

    plane = xSpace * x + ySpace * y + z
    rays = plane - z * D
    norm = rays.reshape(-1, 3).norm(dim=1).view(-1, px, px, 1)
    return rays / norm

def upsample(t, scale):
    return t.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)

def multiscale(axes, frustum, latents, sdf, dtype, threshold):
    batch = axes.shape[0]
    terminal = torch.zeros(batch, 1, 1, 1, dtype=dtype, device=axes.device)
    rayMask = torch.ones_like(terminal, dtype=torch.bool).squeeze(3)
    origin = frustum.cameraD * axes[:, 2][:, None, None, :]
    latents = latents[:, None, None, :]
    size = 1

    first = True
    for scale, step, near, far, sphereMask in frustum:
        if not first:
            # TODO: move this into raymarch()
            # a geometric bound for whether a super-ray could have subrays that intersected the object
            k = (maskedNear + distances) * 2 * np.tan(frustum.fov / (2. * size))
            k = k.squeeze()
            # this is suspicious because it uses a step size which is not actually used by sphere tracing
            # also the 0.5 at the end is empirical
            bound = torch.sqrt(2. * k ** 2. + (1.0 / step) ** 2) * 0.5
            del k

            # subdivide the rays that pass close to the object
            rayMask[rayMask] = minValues <= bound
            del bound
            del minValues
        first = False

        near = near.expand(batch, -1, -1)
        far = far.expand(batch, -1, -1)
        size *= scale
        terminal = upsample(terminal, scale)
        rayMask = upsample(rayMask, scale)
        expandedLatents = latents.expand(-1, size, size, -1)
        expandedOrigin = origin.expand(-1, size, size, -1)
        rays = makeRays(axes, size, frustum.cameraD, frustum.edge, dtype=dtype)

        # terminate rays that don't intersect the unit sphere
        rayMask = torch.logical_and(rayMask, sphereMask)
        maskedNear = near[rayMask]
        planes = torch.stack([maskedNear, far[rayMask]])

        minValues, distances = sphereTrace(rays[rayMask],
                                        expandedOrigin[rayMask],
                                        planes,
                                        expandedLatents[rayMask],
                                        sdf,
                                        steps=step,
                                        threshold=threshold,
                                        dtype=dtype)

        terminal[rayMask] = (maskedNear + distances).unsqueeze(1)

    points = origin + terminal * rays

    # use the lowest-resolution mask, because the high res mask includes unsampled rays
    sphereMask = upsample(frustum.mask[0], frustum.imageSize // frustum.scales[0]).expand(batch, -1, -1)
    return points[sphereMask], expandedLatents[sphereMask], sphereMask

def sphereTrace(rays, origin, planes, latents, sdf, steps, threshold, dtype):
    device = rays.device
    minValues = 5.0 * torch.ones(rays.shape[0], device=device, dtype=dtype)
    mask = torch.ones_like(minValues, dtype=torch.bool)

    # all sampled points are on concentric circles unless starting distance is random
    distance = torch.ones_like(minValues) * (2.0 / steps)
    minDistances = torch.ones_like(minValues) * distance

    # start from 1, because the initial critical points are the 0-idx
    for i in range(1, steps):
        # march along each ray
        # compute the next target point
        targets = (origin + rays * (planes[0] + distance).unsqueeze(1))[mask]

        # stop if all rays have terminated
        if targets.shape[0] == 0:
            break

        # evaluate the targets
        with autocast():
            values = sdf.forward_inplace(targets, latents, mask).squeeze(1).type(dtype)

        del targets

        # TODO: this is the bottleneck
        # Boolean indexing produces a variably-sized result, which causes an
        # expensive CPU-GPU sync. But all of these masks work together to limit the
        # number of SDF queries, which is also very expensive.
        minMask = values < minValues[mask]
        updateMask = torch.zeros_like(mask)
        updateMask[mask] = minMask

        # update min and argmin
        minValues[updateMask] = values[minMask]
        minDistances[updateMask] = distance[updateMask]


        # terminate all rays that intersect the surface (negated)
        intersectMask = values > threshold

        # terminate rays that exit the unit sphere on the next step (again negated)
        maskedPlanes = planes[mask.expand(2, -1)].view(2, -1)
        exitMask = (maskedPlanes[0] + distance[mask]) < maskedPlanes[1]

        distance[mask] += values
        mask[mask] = torch.logical_and(intersectMask, exitMask)

    return minValues, minDistances
