import torch
import torch.nn as nn
from .sdfNetwork import SDFNetwork

class Generator(nn.Module):
    def __init__(self, weightNorm=False):
        self.sdf = SDFNetwork(weightNorm=weightNorm)

    # DFR section 3.5 (ray integral)
    def rayIntegral(self, x, rayCount):
        # the initial coarse sampling step is done without grad
        with torch.no_grad():
            # x has shape [rayCount * sampleCount, 3 + latentSize]
            # for a single object, sample a batch of points
            values = self.sdf(x)

            # samples has shape [rayCount * sampleCount]
            # reshape into [rayCount, sampleCount]
            rays = torch.reshape(values, (rayCount, -1))

            # find the minimum sampled value over each ray
            # -5e-11 is the minimum depth that is considered an intersection
            clamped = torch.clamp(rays, -5e-11, 10.0)
            minIdx = torch.argmin(clamped, dim=1)

            # pull useful points from sample points
            critPoints = torch.reshape(x, (rayCount, -1, -1))[:, minIdx]

            # flatten again
            flattenedPoints = torch.flatten(critPoints, 0, 1)

        # now, with gradient, sample the useful points
        # TODO: compute surface normals here
        return self.sdf(flattenedPoints)

    # DFR section 3.6 (soft shading)
    def shader(self, values, k=10.0):
        return 1.0 / (1.0 + torch.exp(-k * values))

    def forward(self, x):
        pass
