import torch

# compute an illumination map based on the angle between the light and the surface normal
def illuminate(light, normals):
    dot = torch.matmul(normals.view(light.shape[0], -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1)
    return (dot + 1.0) / 2.0

# given SDF values, normals, and texture, construct an image
def shade(data, light, sphereMask, sharpness):
    illum = illuminate(light, data.normals)
    illum[data.values > 0.] = 1.

    shading = torch.zeros(*sphereMask.shape, 3, device=sphereMask.device)
    # TODO: is the float() necessary?
    shading[sphereMask] = (illum * data.textures).float()

    valueMap = torch.ones(*sphereMask.shape, 1, device=sphereMask.device)
    valueMap[sphereMask] = data.values.float()

    valueMap = valueMap.permute(0, 3, 1, 2)
    shading = shading.permute(0, 3, 1, 2)

    opacity = torch.exp(-sharpness * valueMap).clamp(max = 1.0)
    return torch.cat([shading * opacity, opacity], dim=1)
