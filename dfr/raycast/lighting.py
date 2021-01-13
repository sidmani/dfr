import torch
import numpy as np

# compute an illumination map based on the angle between the light and the surface normal
def illuminate(light, normals):
    dot = torch.matmul(normals.view(light.shape[0], -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1)
    return (dot + 1.0) / 2.0

# given SDF values, normals, and texture, construct an image
def shade(data, light, sphereMask, sigma, threshold):
    illum = illuminate(light, data.normals)
    illum[data.values > threshold] = 1.

    shading = torch.zeros(*sphereMask.shape, 3, device=sphereMask.device)
    # TODO: is the float() necessary?
    shading[sphereMask] = (illum * data.textures).float()

    valueMap = torch.ones(*sphereMask.shape, 1, device=sphereMask.device)
    valueMap[sphereMask] = data.values.float()

    valueMap = valueMap.permute(0, 3, 1, 2)
    shading = shading.permute(0, 3, 1, 2)

    if sigma > 0:
        surfaceMask = (valueMap < threshold).float()
        fuzz = (1 - torch.erf(valueMap / (np.sqrt(2) * sigma * 2)))
        surface = 1 - valueMap
        opacity = fuzz * (1 - surfaceMask) + surface * surfaceMask


        # upper = torch.threshold((1 - valueMap + threshold), 1, 0).clamp(0, 1)
        # sigma is normalized to real distances, so multiply by the resolution to get pixel distances
        # upper = blur(upper, sigma * sphereMask.shape[2])

        # The gaussian blur falls off as erf(x), and the blur is symmetric around 0
        # So we have the values follow erf(x), for value > 0.5
        # multiply sigma by 2 since the unit sphere fills the fov
        # lower = 0.5 * (1 - torch.erf(valueMap / (np.sqrt(2) * sigma * 2)))
        # mask = (valueMap > 0).float()
        # opacity = upper
        # opacity = (1 - mask) * upper + mask * lower
    else:
        opacity = torch.threshold((1 - valueMap + threshold), 1, 0).clamp(0, 1)

    return torch.cat([shading * opacity, opacity], dim=1)
