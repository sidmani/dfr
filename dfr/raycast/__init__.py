import torch
from .ray import sampleRays, findIntersection
from .frustum import sphereToRect

def raycast(phis,
            thetas,
            latents,
            frustum,
            sdf,
            texture,
            raySamples,
            bgNoise=True):
    device = phis.device
    batch = phis.shape[0]

    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        targets = sampleRays(phis, thetas, frustum, raySamples)
        critPoints = findIntersection(latents, targets, sdf).view(-1, 3)
    critPoints.requires_grad = True

    sampleCount = critPoints.shape[0] // batch
    expandedLatents = torch.repeat_interleave(latents, sampleCount, dim=0)
    x = torch.cat([critPoints, expandedLatents], dim=1)

    # sample the critical points with autograd enabled
    values = sdf(x)

    # TODO: only sample texture for hit points
    textures = texture(x)

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones(values.shape, device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    # background is random noise
    if bgNoise:
        result = torch.normal(0.5, 0.1, size=(batch, *frustum.mask.shape, 3), device=device)
    else:
        result = torch.zeros(batch, *frustum.mask.shape, 3, device=device)

    # [batch, # hit sphere, 1]
    values = values.view(batch, -1, 1)
    textures = textures.view(batch, -1, 3)
    hitMask = values.squeeze(2) <= 0.0

    # create composite mask: select rays that hit unit sphere and object
    mask = torch.zeros(*result.shape[:3], dtype=torch.bool)
    mask[:, frustum.mask] = hitMask

    # apply the sampled texture to the hit points
    result[mask] = textures[hitMask]

    return result.permute(0, 3, 1, 2), normals
