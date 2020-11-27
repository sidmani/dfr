import torch
from .ray import iterativeIntersection, rotateFrustum

def raycast(phis,
            thetas,
            latents,
            frustum,
            sdf,
            raySamples):
    batch = phis.shape[0]

    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        rays, cameraLoc = rotateFrustum(phis, thetas, frustum, jitter=True)
        critPoints, hitMask = iterativeIntersection(rays, frustum, cameraLoc, latents, sdf, steps=raySamples)
    critPoints.requires_grad = True
    cameraLoc.requires_grad = True
    notHitMask = ~hitMask

    # sample the critical points with autograd enabled
    expandedLatents = torch.repeat_interleave(latents, critPoints.shape[1], dim=0)
    critPoints = critPoints.view(-1, 3)
    values, textures = sdf(critPoints, expandedLatents)
    values = values.view(batch, -1)
    textures = textures.view(batch, -1, 3)

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    unitNormals = normals / (normals.norm(dim=1).unsqueeze(1) + 1e-5)

    # light is directed from camera
    light = cameraLoc / frustum.cameraD

    # scale dot product from [-1, 1] to [0, 1]
    illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(batch, 1, 3, 1)).view(batch, -1, 1) + 1.0) / 2.0
    illum[notHitMask] = 1.0

    result = torch.zeros(batch, 4, *frustum.mask.shape, device=phis.device)
    opacityMask = torch.ones_like(values)
    opacityMask[notHitMask] = torch.exp(-10.0 * values[notHitMask])
    result[:, :3, frustum.mask] = (opacityMask.unsqueeze(2) * illum * textures).permute(0, 2, 1)
    result[:, 3, frustum.mask] = opacityMask

    return result, normals
