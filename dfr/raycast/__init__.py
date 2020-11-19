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
    hitMask = values <= 0.0
    notHitMask = ~hitMask

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones(values.shape, device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    unitNormals = normals / normals.norm(dim=1).unsqueeze(1)

    # light is directed from camera
    light = cameraLoc / frustum.cameraD

    # scale dot product from [-1, 1] to [0, 1]
    illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(batch, 1, 3, 1)).view(batch, -1, 1) + 1.0) / 2.0
    illum[notHitMask] = 1.0

    result = torch.zeros(batch, 4, *frustum.mask.shape, device=device)
    opacityMask = torch.ones(*values.shape, device=device)
    opacityMask[notHitMask] = torch.exp(-10.0 * values[notHitMask])
    result[:, :3, frustum.mask] = (opacityMask.unsqueeze(2) * illum * textures).permute(0, 2, 1)

    # silhouette = torch.ones(*values.shape, device=device)
    # silhouette[notHitMask] = torch.exp(-10.0 * values[notHitMask])
    result[:, 3, frustum.mask] = opacityMask

    return result, normals
