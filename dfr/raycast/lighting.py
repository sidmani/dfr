import torch
import numpy as np
from ..image import blur

# a step function with modified gradients to allow differentiability
# class SigmaStep(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return torch.heaviside(input, torch.zeros_like(input))
#         # return torch.exp(input)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         # grad = -10 * torch.exp(-10 * input)
#         grad = torch.exp(input)
#         # grad_input = grad_output.clone()
#         # grad_input[input < 0] = 1.
#         # grad_input[input >= 0] = 0
#         return grad * grad_output

# compute an illumination map based on the angle between the light and the surface normal
def illuminate(light, normals):
    dot = torch.matmul(normals.view(light.shape[0], -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1)
    return (dot + 1.0) / 2.0

def unmask(values, sphereMask):
    valueMap = torch.ones(*sphereMask.shape, values.shape[1], device=sphereMask.device)
    valueMap[sphereMask] = values
    return valueMap.permute(0, 3, 1, 2)

# given SDF values, normals, and texture, construct an image
def shade(data, light, sphereMask, sigma):
    illum = illuminate(light, data.normals)

    valueMap = unmask(data.values.float(), sphereMask)
    colorMap = unmask(data.textures.float(), sphereMask)
    illumMap = unmask(illum.float(), sphereMask)

    # the illumination value isn't well defined outside the surface, and can mess up the gradients
    # so just set it to one.
    illumMap[valueMap > 0] = 1.0

    if sigma > 0:
        surfaceMask = (valueMap < 0).float()
        fuzz = (1 - torch.erf(valueMap / (np.sqrt(2) * sigma * 2)))
        surface = torch.threshold((1 - valueMap), 1, 0).clamp(0, 1)
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
        opacity = torch.threshold((1 - valueMap), 1, 0).clamp(0, 1)

    return torch.cat([illumMap * colorMap * opacity, opacity], dim=1)
