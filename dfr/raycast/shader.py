import torch

# DFR section 3.5 (ray integral)
def searchRays(latents, targets, sdf, epsilon):
    # targets has shape [batch, rays, samples, 3]
    # latents has shape [batch, latentSize]
    # evaluate SDF
    values = sdf(targets.view(-1, 3), latents).view(*targets.shape[:3])

    # find the minimum sampled value over each ray
    # epsilon is the minimum depth that is considered an intersection
    clamped = torch.clamp(values, min=-epsilon)

    # gather critical points from sample points
    # see https://stackoverflow.com/questions/53471716/index-pytorch-4d-tensor-by-values-in-2d-tensor
    minIdx = torch.argmin(clamped, dim=2)[..., None, None].expand(-1, -1, 1, 3)
    return torch.gather(targets, dim=2, index=minIdx).squeeze(2)

def fastRayIntegral(latents, targets, sdf, epsilon):
    with torch.no_grad():
        critPoints = searchRays(latents, targets, sdf, epsilon)

    # now, with gradient, sample the useful points
    # TODO: compute normals here
    return sdf(critPoints.view(-1, 3), latents).view(*targets.shape[:2])

def shade(values, k=10.0, j=9.0):
    return 1.0 / (1.0 + j * torch.exp(-k * values))
