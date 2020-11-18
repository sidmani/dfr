import torch
from .ray import sampleRays, findIntersection
from .frustum import sphereToRect

def raycast(phis,
            thetas,
            latents,
            frustum,
            sdf,
            raySamples,
            bgNoise=True):
    device = phis.device
    batch = phis.shape[0]

    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        targets, cameraLoc = sampleRays(phis, thetas, frustum, raySamples)
        critPoints = findIntersection(latents, targets, sdf).view(-1, 3)
    critPoints.requires_grad = True
    cameraLoc.requires_grad = True

    # sample the critical points with autograd enabled
    values, textures = sdf(critPoints, latents)
    values = values.view(batch, -1)
    textures = textures.view(batch, -1, 3)

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones(values.shape, device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    # unitNormals = normals / normals.norm(dim=1).unsqueeze(1)

    # light is directed from camera
    # light = cameraLoc / frustum.cameraD

    # scale dot product from [-1, 1] to [0, 1]
    # illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(batch, 1, 3, 1)).view(batch, -1, 1) + 1.0) / 2.0

    # background is random noise
    if bgNoise:
        result = torch.normal(0.5, 0.1, size=(batch, *frustum.mask.shape, 3), device=device)
    else:
        result = torch.zeros(batch, *frustum.mask.shape, 3, device=device)

    notHitMask = values > 0.0
    opacityMask = torch.ones_like(values, device=device)
    opacityMask[notHitMask] = torch.exp(-10.0 * values[notHitMask])
    opacityMask = opacityMask.unsqueeze(2)
    result[:, frustum.mask] = opacityMask * textures + (1.0 - opacityMask) * result[:, frustum.mask]

    # apply the sampled texture to the hit points
    return result.permute(0, 3, 1, 2), normals
