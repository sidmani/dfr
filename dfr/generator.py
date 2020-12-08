import torch
import torch.nn as nn
import numpy as np
from .raycast import raycast
from torch.cuda.amp import autocast

class Generator(nn.Module):
    def __init__(self, sdf, frustum, hparams):
        super().__init__()
        self.sdf = sdf
        self.frustum = frustum
        self.hparams = hparams

    def forward(self, phis, thetas, latents, gradScaler):
        return raycast(phis, thetas, self.frustum, latents, self.sdf, gradScaler)

    def sample(self, batchSize, gradScaler, phi=np.pi/6.0, device=None):
        with torch.no_grad(), autocast():
            # elevation angle: uniform pi/12 <= phi <= 3pi/12
            # phis = (torch.rand(batchSize, device=device) + 0.5) * (np.pi / 6.0)
            phis = torch.ones(batchSize, device=device) * phi
            # azimuthal angle: uniform 0 <= theta < 2pi
            thetas = torch.rand_like(phis) * (2.0 * np.pi)
            # latents with mean 0, stddev >0.3
            # DFR uses sigma=sqrt(0.33)=0.57
            # SALD/DeepSDF use much smaller stddev, but the latent space is optimized in those models
            z = torch.normal(
                    mean=0.0,
                    std=self.hparams.latentStd,
                    size=(batchSize, self.hparams.latentSize),
                    device=device)

        return self.forward(phis, thetas, z, gradScaler)

    def sample_like(self, other, gradScaler):
        return self.sample(other.shape[0], gradScaler, device=other.device)
