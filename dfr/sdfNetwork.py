import torch
import torch.nn as nn
import numpy as np
from .geometricInit import geometricInit

class SDFNetwork(nn.Module):
    def __init__(self, hparams, width=512):
        super().__init__()
        assert width > hparams.latentSize + 3

        self.layers = nn.ModuleList([
            nn.Linear(hparams.latentSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (hparams.latentSize + 3)),
            # skip connection from input: latent + x (into 5th layer)
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        # SAL geometric initialization
        # geometricInit(self.layers)

        self.hparams = hparams
        if hparams.weightNorm:
            for i in range(8):
                self.layers[i] = nn.utils.weight_norm(self.layers[i])

        # DeepSDF uses ReLU, SALD uses Softplus
        self.activation = nn.ReLU()

    def forward(self, pts, latents):
        # first 4 layers only deal with latents
        r = latents
        for i in range(4):
            r = self.activation(self.layers[i](r))

        # skip connection
        # per SAL supplementary: divide by sqrt(2) to normalize
        sampleCount = pts.shape[0] // latents.shape[0]
        processed = torch.repeat_interleave(r, sampleCount, dim=0)

        # may be a way to create this in parent
        expandedLatents = torch.repeat_interleave(latents, sampleCount, dim=0)

        # TODO: layer idx 4 can be split into separate matrices and cached
        # expected gain ~7%
        r = torch.cat([pts, expandedLatents, processed], dim=1) / np.sqrt(2)
        for i in range(3):
            r = self.activation(self.layers[i + 4](r))

        return torch.tanh(self.layers[7](r))
