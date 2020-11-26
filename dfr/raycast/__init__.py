import torch
from .ray import intersection, rotateFrustum

def raycast(phis, thetas, latents, frustum, sdf):
    batch = phis.shape[0]
    img = (frustum.imageSize, frustum.imageSize)

    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        rays, cameraLoc = rotateFrustum(phis, thetas, frustum, jitter=True)
        critPoints = intersection(rays, frustum, cameraLoc, latents, sdf)
    critPoints.requires_grad = True
    cameraLoc.requires_grad = True

    # sample the critical points with autograd enabled
    expandedLatents = torch.repeat_interleave(latents, frustum.imageSize ** 2, dim=0)
    critPoints = critPoints.reshape(-1, 3)
    values, textures = sdf(critPoints, expandedLatents)

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    # need epsilon in denominator for numerical stability (otherwise training destabilizes)
    unitNormals = normals / (normals.norm(dim=1).unsqueeze(1) + 1e-5)

    # light is directed from camera
    light = cameraLoc / frustum.cameraD
    notHitMask = values > 0.0

    # scale dot product from [-1, 1] to [0, 1]
    illum = (torch.matmul(unitNormals.view(batch, -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1) + 1.0) / 2.0
    illum[notHitMask] = 1.0

    result = torch.zeros(batch, 4, *img, device=phis.device)
    opacityMask = torch.ones_like(values)
    opacityMask[notHitMask] = torch.exp(-10.0 * values[notHitMask])

    result[:, :3] = (opacityMask * textures * illum).view(batch, *img, 3).permute(0, 3, 1, 2)
    result[:, 3] = opacityMask.view(batch, *img)

    # values = torch.clamp(values.view(batch, *img), min=0)
    # result[:, 4] = values / (torch.amax(values, dim=(1, 2))[:, None, None] + 1e-5)

    return result, normals
