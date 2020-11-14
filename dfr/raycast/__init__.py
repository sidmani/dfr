import torch
from .ray import sampleRays, findIntersection
from .frustum import sphereToRect

def raycast(phis, thetas, latents, frustum, sdf, texture, raySamples):
    device = phis.device

    # autograd isn't needed here; no backprop to the camera position
    with torch.no_grad():
        targets = sampleRays(phis, thetas, frustum, raySamples)
        critPoints = findIntersection(latents, targets, sdf).view(-1, 3)
    critPoints.requires_grad = True

    sampleCount = critPoints.shape[0] // latents.shape[0]
    expandedLatents = torch.repeat_interleave(latents, sampleCount, dim=0)
    x = torch.cat([critPoints, expandedLatents], dim=1)

    # sample the critical points with autograd enabled
    values = sdf(x)
    textures = texture(x)

    # compute normals
    normals = torch.autograd.grad(outputs=values,
                inputs=critPoints,
                grad_outputs=torch.ones(values.shape, device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

    shaded = shade(values, textures, normals)

    result = torch.zeros(phis.shape[0],
                         frustum.imageSize,
                         frustum.imageSize,
                         3,
                         device=device)
    result[:, frustum.mask] = shaded.view(result.shape[0], -1, 3)
    return result, normals

def shade(values, texture, normals, fuzz=15.0):
    # lightDir = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1)
    # unitNormals = (normals / normals.norm(dim=1).unsqueeze(1)).view(-1, 1, 3)
    # lightFactor = torch.matmul(unitNormals, lightDir).squeeze(1)
    # shade only the points that intersect the surface with the sampled color
    # result = texture * lightFactor
    result = torch.empty(texture.shape, device=values.device)
    hits = values.squeeze() <= 0.0
    notHits = ~hits

    result[hits] = texture[hits]
    result[notHits] = texture[notHits] * torch.exp(-fuzz * values[notHits])
    return result
