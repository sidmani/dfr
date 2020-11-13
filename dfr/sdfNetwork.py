import torch
import torch.nn as nn
import numpy as np
from .geometricInit import geometricInit

class SDFNetwork(nn.Module):
    def __init__(self, hparams, width=512):
        super().__init__()
        assert width > hparams.latentSize + 3

        self.sdfLayers = nn.ModuleList([
            nn.Linear(hparams.latentSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (hparams.latentSize + 3)),
            # skip connection from input: latent + x (into 5th layer)
            nn.Linear(width, width),
            nn.Linear(width, width),
            # branch here for texture
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        self.textureLayers = nn.ModuleList([
            nn.Linear(width, width),
            nn.Linear(width, 3),
        ])

        # SAL geometric initialization
        geometricInit(self.sdfLayers)

        for layer in self.textureLayers:
            nn.init.normal_(layer.weight)

        self.hparams = hparams
        if hparams.weightNorm:
            for i in range(8):
                self.sdfLayers[i] = nn.utils.weight_norm(self.sdfLayers[i])

            for i in range(2):
                self.textureLayers[i] = nn.utils.weight_norm(self.textureLayers[i])

        # DeepSDF uses ReLU, SALD uses Softplus
        self.activation = nn.ReLU()

    def forward(self, pts, latents, geometryOnly=False):
        # first 4 layers only deal with latents
        r = latents
        for i in range(4):
            r = self.activation(self.sdfLayers[i](r))

        # skip connection
        # per SAL supplementary: divide by sqrt(2) to normalize
        sampleCount = pts.shape[0] // latents.shape[0]
        processed = torch.repeat_interleave(r, sampleCount, dim=0)

        # may be a way to create this in parent
        expandedLatents = torch.repeat_interleave(latents, sampleCount, dim=0)

        # TODO: layer idx 4 can be split into separate matrices and cached
        # expected gain ~7%
        r = torch.cat([pts, expandedLatents, processed], dim=1) / np.sqrt(2)
        for i in range(2):
            r = self.activation(self.sdfLayers[i + 4](r))

        # SDF portion
        sdf = self.activation(self.sdfLayers[6](r))
        sdf = torch.tanh(self.sdfLayers[7](sdf))
        if geometryOnly:
            return sdf

        # Texture portion
        texture = self.activation(self.textureLayers[0](r))
        texture = torch.sigmoid(self.textureLayers[1](texture))

        return sdf, texture
