import torch
import torch.nn as nn
import numpy as np
from .raycast import raycast, shade

class Generator(nn.Module):
    def __init__(self, sdf, texture, frustum, hparams):
        super().__init__()
        self.sdf = sdf
        self.texture = texture
        self.frustum = frustum
        self.hparams = hparams

    def forward(self, phis, thetas, latents):
        values, normals = raycast(
                phis,
                thetas,
                latents,
                self.frustum,
                self.sdf,
                self.hparams.raySamples)

    def sample(self, batchSize, phi=np.pi / 6.0, device=None):
        # elevation angle: phi = pi/6
        phis = torch.ones(batchSize, device=device) * phi
        # azimuthal angle: 0 <= theta < 2pi
        thetas = torch.rand(batchSize, device=device) * (2.0 * np.pi)
        # latents with mean 0, sigma 1e-4 (per SALD)
        # DFR uses sigma=sqrt(0.33), but that's a different architecture (OccNet)
        z = torch.normal(
                mean=0.0,
                std=1e-2,
                size=(batchSize, self.hparams.latentSize),
                device=device)

        return self.forward(z, phis, thetas)
