import torch
import numpy as np

def rotateAxes(phis, thetas):
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    cos_phi = torch.cos(phis)
    sin_phi = torch.sin(phis)

    # composition of 2 transforms: rotate theta first, then phi
    return torch.stack([
        torch.stack([cos_theta, torch.zeros_like(phis), -sin_theta]),
        torch.stack([-sin_phi * sin_theta, cos_phi, -cos_theta * sin_phi]),
        torch.stack([cos_phi * sin_theta, sin_phi, cos_phi * cos_theta]),
    ]).permute(2, 0, 1)

def makeRays(axes, px, D, edge):
    # TODO: offset
    xSpace = torch.linspace(-edge, edge, steps=px, device=axes.device).repeat(px, 1)[None, :, :, None]
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

def multiscale(axes, frustum, latents, sdf):
    batch = axes.shape[0]
    terminal = torch.zeros(batch, 1, 1, 1, device=axes.device)
    rayMask = torch.ones_like(terminal, dtype=torch.bool).squeeze(3)
    origin = frustum.cameraD * axes[:, 2][:, None, None, :]
    latents = latents[:, None, None, :]
    size = 1

    first = True
    for scale, step, near, far, sphereMask in frustum:
        if not first:
            # a geometric bound for whether a super-ray could have subrays that intersected the object
            k = (maskedNear + distances) * 2 * np.tan(frustum.fov / (2 * float(size)))
            k = k.squeeze()
            bound = torch.sqrt(2. * k ** 2. + (1.0 / step) ** 2)

            # subdivide the rays that pass close to the object
            rayMask[rayMask] = minValues <= bound
        first = False

        near = near.expand(batch, -1, -1)
        far = far.expand(batch, -1, -1)

        size *= scale
        terminal = upsample(terminal, scale)
        rayMask = upsample(rayMask, scale)
        expandedLatents = latents.expand(-1, size, size, -1)
        expandedOrigin = origin.expand(-1, size, size, -1)
        rays = makeRays(axes, size, frustum.cameraD, frustum.edge)

        # terminate rays that don't intersect the unit sphere
        rayMask = torch.logical_and(rayMask, sphereMask)
        maskedNear = near[rayMask]

        minValues, distances = raymarch(rays[rayMask],
                                        expandedOrigin[rayMask],
                                        maskedNear,
                                        far[rayMask],
                                        expandedLatents[rayMask],
                                        sdf,
                                        steps=step,
                                        stepSize=2.0 / step,
                                        refine=size == frustum.imageSize)

        terminal[rayMask] = (maskedNear + distances).unsqueeze(1)

    points = origin + terminal * rays
    # use the lowest-resolution mask, because the high res mask includes unsampled rays
    sphereMask = upsample(frustum.mask[0], frustum.imageSize // frustum.scales[0]).expand(batch, -1, -1)
    return points[sphereMask], expandedLatents[sphereMask], sphereMask

def raymarch(rays, origin, near, far, latents, sdf, steps, stepSize, refine=False):
    device = rays.device
    mask = torch.ones(rays.shape[0], dtype=torch.bool, device=device)

    minValues = 5.0 * torch.ones_like(mask, dtype=torch.float)

    # all sampled points are on concentric circles unless starting distance is random
    distance = (1.0 + torch.rand(1, device=device)) * stepSize
    minDistances = torch.ones_like(minValues) * distance

    # start from 1, because the initial critical points are the 0-idx
    for i in range(1, steps):
        # march along each ray
        distance += stepSize
        # compute the next target point
        targets = origin[mask] + rays[mask] * (near[mask].unsqueeze(1) + distance)

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
        minDistances[updateMask] = distance

        # terminate all rays that intersect the surface (negated)
        intersectMask = values > 0.0
        # terminate rays that exit the unit sphere on the next step (again negated)
        exitMask = (near[mask] + distance + stepSize) < far[mask]
        mask[mask] = torch.logical_and(intersectMask, exitMask)

    if refine:
        # refine rays that intersect the surface
        # step 1 back, then move forward in smaller increments
        distance = minDistances - stepSize
        mask = minValues < 0.0
        minValues = 5.0 * torch.ones_like(minValues, dtype=torch.float)
        n = 8
        stepSize = stepSize / n
        for i in range(n + 1):
            distance[mask] += stepSize
            targets = origin[mask] + rays[mask] * (near[mask] + distance[mask]).unsqueeze(1)

            if targets.shape[0] == 0:
                break

            # evaluate the targets
            values = sdf(targets, latents[mask], geomOnly=True).squeeze(1)
            minMask = values < minValues[mask]
            updateMask = torch.zeros_like(mask)
            updateMask[mask] = minMask

            # update min and argmin
            minValues[updateMask] = values[minMask]
            minDistances[updateMask] = distance[updateMask]

            # terminate all rays that intersect the surface (negated)
            mask[mask] = values > 0.0
    return minValues, minDistances
