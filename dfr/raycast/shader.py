import torch

# DFR section 3.5 (ray integral)
def searchRays(latents, targets, sdf, epsilon):
    # targets has shape [batch, rays, samples, 3]
    # latents has shape [batch, latentSize]
    # flatten into 2d tensor
    flattenedPoints = targets.view(-1, 3)
    latentsTiled = torch.repeat_interleave(latents, targets.shape[1] * targets.shape[2], dim=0)

    # concat latents
    batch = torch.cat([flattenedPoints, latentsTiled], dim=1)

    # evaluate SDF
    values = sdf(batch).view(*targets.shape[:3])
    print(values)

    # find the minimum sampled value over each ray
    # epsilon is the minimum depth that is considered an intersection
    clamped = torch.clamp(values, min=-epsilon)

    # gather critical points from sample points
    # see https://stackoverflow.com/questions/53471716/index-pytorch-4d-tensor-by-values-in-2d-tensor
    minIdx = torch.argmin(clamped, dim=2)[..., None, None].expand(-1, -1, 1, 3)
    return torch.gather(targets, dim=2, index=minIdx).squeeze(2)

def raycast(latents, targets, sdf, epsilon):
    with torch.no_grad():
        critPoints = searchRays(latents, targets, sdf, epsilon)

    flattenedPoints = critPoints.view(-1, 3)
    latentsTiled = torch.repeat_interleave(latents, targets.shape[1], dim=0)
    batch = torch.cat([flattenedPoints, latentsTiled], dim=1)

    # now, with gradient, sample the useful points
    # TODO: compute normals here
    return sdf(batch).view(*targets.shape[:2])
