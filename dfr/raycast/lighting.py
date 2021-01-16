import torch
import numpy as np
from ..image import blur

# a step function with modified gradients to allow differentiability
class SigmaStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.heaviside(input, torch.zeros_like(input))
        # return torch.exp(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad = -10 * torch.exp(-10 * input)
        grad = torch.exp(input)
        # grad_input = grad_output.clone()
        # grad_input[input < 0] = 1.
        # grad_input[input >= 0] = 0
        return grad * grad_output

# compute an illumination map based on the angle between the light and the surface normal
def illuminate(light, normals):
    dot = torch.matmul(normals.view(light.shape[0], -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1)
    return (dot + 1.0) / 2.0

def unmask(values, sphereMask):
    valueMap = torch.ones(*sphereMask.shape, values.shape[1], device=sphereMask.device)
    valueMap[sphereMask] = values
    return valueMap.permute(0, 3, 1, 2)

# given SDF values, normals, and texture, construct an image
def shade(data, light, sphereMask, sigma, threshold):
    illum = illuminate(light, data.normals)

    valueMap = unmask(data.values.float(), sphereMask)
    colorMap = unmask(data.textures.float(), sphereMask)
    illumMap = unmask(illum.float(), sphereMask)
    illumMap[valueMap > threshold] = 1.0

    opacity = SigmaStep.apply(-10 * (valueMap - threshold))
    # opacity = torch.sigmoid(-valueMap + threshold)
    # opacity = torch.exp(-10 * (valueMap - threshold))
    return torch.cat([illumMap * colorMap * opacity, opacity], dim=1)
