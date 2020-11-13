import torch

# DFR section 3.5 (ray integral)
def searchRays(latents, targets, sdf, epsilon):
    # targets has shape [batch, rays, samples, 3]
    # latents has shape [batch, latentSize]
    # evaluate SDF
    values = sdf(targets.view(-1, 3), latents, geometryOnly=True).view(*targets.shape[:3])

    # find the minimum sampled value over each ray
    # epsilon is the minimum depth that is considered an intersection
    clamped = torch.clamp(values, min=-epsilon)

    # gather critical points from sample points
    # see https://stackoverflow.com/questions/53471716/index-pytorch-4d-tensor-by-values-in-2d-tensor
    minIdx = torch.argmin(clamped, dim=2)[..., None, None].expand(-1, -1, 1, 3)
    return torch.gather(targets, dim=2, index=minIdx).squeeze(2)

def fastRayIntegral(latents, targets, sdf, epsilon):
    device = latents.device

    with torch.no_grad():
        critPoints = searchRays(latents, targets, sdf, epsilon).view(-1, 3)

    critPoints.requires_grad = True

    # now, with gradient, sample the useful points
    values, texture = sdf(critPoints, latents)

    # compute the normals at each point
    normals = torch.autograd.grad(outputs=values,
                                  inputs=critPoints,
                                  grad_outputs=torch.ones(values.shape, device=device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]

    return values, texture, normals

def shadeUniform(values, k=40.0, j=15.0):
    return 1.0 / (1.0 + j * torch.exp(-k * values))

def shade(values, texture, normals, fuzz=15.0):
    # print(normals.shape)
    # lightDir = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1)
    # unitNormals = (normals / normals.norm(dim=1).unsqueeze(1)).view(-1, 1, 3)
    # lightFactor = torch.matmul(unitNormals, lightDir).squeeze(1)
    # shade only the points that intersect the surface with the sampled color
    # result = texture * lightFactor
    result = torch.empty(texture.shape)
    hits = values.squeeze(1) <= 0.0
    notHits = ~hits

    result[hits] = texture[hits]
    result[notHits] = texture[notHits] * torch.exp(-fuzz * values[notHits])
    return result
