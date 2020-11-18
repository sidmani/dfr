import torch
import torch.nn as nn
import numpy as np
from .raycast import raycast

class Generator(nn.Module):
    def __init__(self, sdf, frustum, hparams):
        super().__init__()
        self.sdf = sdf
        self.frustum = frustum
        self.hparams = hparams

    def forward(self, phis, thetas, latents):
        return raycast(
                phis,
                thetas,
                latents,
                self.frustum,
                self.sdf,
                self.hparams.raySamples)

    def sample(self, batchSize, phi=np.pi / 6.0, device=None):
        # elevation angle: phi = pi/6
        phis = torch.ones(batchSize, device=device) * phi
        # azimuthal angle: uniform 0 <= theta < 2pi
        thetas = torch.rand(batchSize, device=device) * (2.0 * np.pi)
        # latents with mean 0, stddev >0.3
        # DFR uses sigma=sqrt(0.33)
        # SALD/DeepSDF use much smaller stddev, but the latent space is optimized in those models
        # empirically need at least 0.1
        z = torch.normal(
                mean=0.0,
                std=self.hparams.latentStd,
                size=(batchSize, self.hparams.latentSize),
                device=device)

        return self.forward(phis, thetas, z)
