import torch
import numpy as np

# compute an illumination map based on the angle between the light and the surface normal
# def illuminate(light, normals):
#     dot = torch.matmul(normals.view(light.shape[0], -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1)
#     return (dot + 1.0) / 2.0

# take values masked to a circle and arrange them in a square image
def unmask(values, mask):
    valueMap = torch.ones(*mask.shape, values.shape[1], device=mask.device)
    # have to cast manually, since AMP doesn't autocast index puts
    valueMap[mask] = values.to(valueMap.dtype)
    return valueMap.permute(0, 3, 1, 2)

# given SDF values, normals, and texture, construct an image
def shade(data, light, sphereMask, sigma):
    # illum = illuminate(light, data.normals)

    valueMap = unmask(data.values, sphereMask)
    colorMap = unmask(data.textures, sphereMask)
    # illumMap = unmask(illum, sphereMask)

    # the illumination value isn't well defined outside the surface, and can mess up the gradients
    # so just set it to one. not clear if this affects discriminator.
    # illumMap[valueMap > 0] = 1.0

    # if sigma > 0:
    surfaceMask = (valueMap < 0).float()
    fuzz = (1 - torch.erf(valueMap / (np.sqrt(2) * sigma * 2)))
    surface = torch.threshold((1 - valueMap), 1, 0).clamp(0, 1)
    opacity = fuzz * (1 - surfaceMask) + surface * surfaceMask
    # else:
    #     opacity = torch.threshold((1 - valueMap), 1, 0).clamp(0, 1)

    # return torch.cat([illumMap * colorMap * opacity, opacity], dim=1)
    return torch.cat([colorMap * opacity, opacity], dim=1)
