import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from .sdfNetwork import SDFNetwork
from .raycast import buildFrustum, enumerateRays

class Generator(pl.LightningModule):
    def __init__(self, weightNorm=False, fov=2*np.pi/3, px=64):
        self.sdf = SDFNetwork(weightNorm=weightNorm)
        self.px = px

        # the frustum calculation has spherical symmetry, so can precompute it
        (self.cameraD,
         self.phiSpace,
         self.thetaSpace,
         self.segmentNear,
         self.segmentFar) = buildFrustum(fov, px, self.device)

    # DFR section 3.5 (ray integral)
    def rayIntegral(self, x, rayCount, epsilon=5e-14):
        # the initial coarse sampling step is done without grad
        with torch.no_grad():
            # x has shape [rayCount * sampleCount, 3 + latentSize]
            # for a single object, sample a batch of points
            values = self.sdf(x)

            # samples has shape [rayCount * sampleCount]
            # reshape into [rayCount, sampleCount]
            rays = torch.reshape(values, (rayCount, -1))

            # find the minimum sampled value over each ray
            # epsilon is the minimum depth that is considered an intersection
            clamped = torch.clamp(rays, min=-epsilon)
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

    # generate an image; x is a batch of noise vectors
    def forward(self, x):
        # sample a random elevation and azimuth for each object
        # elevation is in [-pi/6, pi/6)
        phis = (torch.rand(x.shape[0]) - 0.5) * (np.pi / 3)

        # azimuth is in [0, 2pi)
        thetas = torch.rand(x.shape[0]) * 2 * np.pi

        # generate rays
        rays = enumerateRays(phis,
                             thetas,
                             self.phiSpace,
                             self.thetaSpace,
                             self.cameraD)
