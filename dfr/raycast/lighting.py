import torch

# compute an illumination map based on the angle between the light and the surface normal
def illuminate(light, normals):
    dot = torch.matmul(normals.view(light.shape[0], -1, 1, 3), light.view(-1, 1, 3, 1)).view(-1, 1)
    return (dot + 1.0) / 2.0

# fuzz the edges for differentiability
def fuzz(values, sharpness):
    return torch.exp(-sharpness * values).clamp(max = 1.0)
