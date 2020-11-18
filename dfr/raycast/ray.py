import torch
from .sample import sampleStratified
from .frustum import sphereToRect

# list the rays given a base frustum and a view angle
def rotateFrustum(phis, thetas, frustum):
    device = phis.device

    zeros = torch.zeros(phis.shape[0], device=device)
    ones = torch.ones(phis.shape[0], device=device)
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    cos_phi = torch.cos(phis)
    sin_phi = torch.sin(phis)

    # create a 3D rotation matrix for each phi, theta pair
    rotation = torch.stack([
        torch.stack([cos_theta, -sin_theta * sin_phi, sin_theta * cos_phi]),
        torch.stack([zeros, cos_phi, sin_phi]),
        torch.stack([-sin_theta, -sin_phi * cos_theta, cos_phi * cos_theta]),
    ]).permute(2, 0, 1).unsqueeze(1)

    # have the cos and sin needed to compute camera location, so let's do it
    cameraLoc = frustum.cameraD * torch.stack([
        cos_phi * sin_theta,
        sin_phi,
        cos_phi * cos_theta,
    ], dim=-1)

    # rotate the frustum and view as [batch, px, px, 3] i.e. a ray for each pixel
    rays = torch.matmul(rotation, frustum.viewField).view(-1, frustum.imageSize, frustum.imageSize, 3)
    return rays, cameraLoc

# scale the rays based on the sample distances
def distributeSamples(rays, samples, cameraLoc):
    # output has shape [batch, selected #, sample count, 3]
    selectedRays = rays.unsqueeze(2).repeat(1, 1, samples.shape[2], 1)
    # [batch, selected #, sample count, 1]
    selectedSamples = samples.unsqueeze(3)

    # sum camera loc with scaled rays to compute sample points
    return cameraLoc.view(-1, 1, 1, 3) + selectedRays * selectedSamples

def sampleRays(phis, thetas, frustum, sampleCount, scheme=sampleStratified):
    # build a rotated frustum for each input angle
    rays, cameraLoc = rotateFrustum(phis, thetas, frustum)
    # cameraLoc = sphereToRect(phis, thetas, frustum.cameraD)

    # uniformly sample distances from the camera in the unit sphere
    # TODO: should sampling be weighted by ray length?
    # unsqueeze because we're using the same sample values for all objects
    samples = scheme(frustum.near, frustum.far, sampleCount).unsqueeze(0)

    # compute the sampling points for each ray that intersects the unit sphere
    return distributeSamples(rays[:, frustum.mask], samples[:, frustum.mask], cameraLoc), cameraLoc

# DFR section 3.5 (ray integral)
# search along each ray for the first intersection or closest point
def findIntersection(latents, targets, sdf, epsilon=1e-10):
    # targets has shape [batch, rays, samples, 3]
    # latents has shape [batch, latentSize]
    # evaluate SDF
    values = sdf(targets.view(-1, 3), latents, geomOnly=True).view(*targets.shape[:3])

    # find the minimum sampled value over each ray
    # epsilon is the minimum depth that is considered an intersection
    clamped = torch.clamp(values, min=-epsilon)

    # gather critical points from sample points
    # see https://stackoverflow.com/questions/53471716/index-pytorch-4d-tensor-by-values-in-2d-tensor
    minIdx = torch.argmin(clamped, dim=2)[..., None, None].expand(-1, -1, 1, 3)
    return torch.gather(targets, dim=2, index=minIdx).squeeze(2)
