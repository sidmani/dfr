import torch
import torch.nn as nn
import numpy as np
from .sdfNetwork import SDFNetwork
from .raycast.frustum import buildFrustum
from .raycast import raycast

class Generator(nn.Module):
    def __init__(self, weightNorm, fov, px, sampleCount, latentSize, device):
        super().__init__()
        self.sampleCount = sampleCount
        self.latentSize = latentSize
        self.sdf = SDFNetwork(weightNorm=weightNorm, latentSize=latentSize)
        # the frustum calculation has spherical symmetry, so can precompute it
        self.frustum = buildFrustum(fov, px, device)
        self.device = device

    def forward(self, latents, phis, thetas):
        return raycast(
                self.sdf,
                latents,
                phis,
                thetas,
                self.frustum,
                self.sampleCount,
                self.device)

    def sample(self, batchSize, phi=np.pi / 6.0):
        # elevation angle: phi = pi/6
        phis = torch.ones(batchSize, device=self.device) * phi
        # azimuthal angle: 0 <= theta < 2pi
        thetas = torch.rand(batchSize, device=self.device) * (2.0 * np.pi)
        # latents with mean 0, variance 0.33
        z = torch.normal(
                mean=0.0,
                std=np.sqrt(0.33),
                size=(batchSize, self.latentSize),
                device=self.device)

        return self(z, phis, thetas)
