import torch
import torch.nn as nn
import numpy as np
from .geometricInit import geometricInit

class SDFNetwork(nn.Module):
    def __init__(self, hparams, width=512):
        super().__init__()
        assert width > hparams.latentSize + 3

        self.sdfLayers = nn.ModuleList([
            nn.Linear(hparams.latentSize + 3, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (hparams.latentSize + 3)),
            # skip connection from input: latent + x (into 5th layer)
            # branch here for texture
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        self.textureLayers = nn.ModuleList([
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 3),
        ])

        # SAL geometric initialization
        geometricInit(self.sdfLayers)

        # SIREN initialization
        for layer in self.textureLayers:
            outDims = float(layer.out_features)
            s = np.sqrt(6/outDims)
            # nn.init.normal_(layer.weight, 0.0, 1.0)
            nn.init.uniform_(layer.weight, -s, s)

        self.hparams = hparams
        if hparams.weightNorm:
            for i in range(8):
                self.sdfLayers[i] = nn.utils.weight_norm(self.sdfLayers[i])

            for i in range(4):
                self.textureLayers[i] = nn.utils.weight_norm(self.textureLayers[i])

        # DeepSDF uses ReLU, SALD uses Softplus
        self.activation = nn.ReLU()

    def forward(self, pts, latents, geometryOnly=False):
        sampleCount = pts.shape[0] // latents.shape[0]
        expandedLatents = torch.repeat_interleave(latents, sampleCount, dim=0)

        r = torch.cat([pts, expandedLatents], dim=1)
        for i in range(4):
            r = self.activation(self.sdfLayers[i](r))

        # skip connection
        # per SAL supplementary: divide by sqrt(2) to normalize
        # TODO: layer idx 4 can be split into separate matrices and cached
        r = torch.cat([pts, expandedLatents, r], dim=1) / np.sqrt(2)

        # SDF portion
        sdf = r
        for i in range(3):
            sdf = self.activation(self.sdfLayers[i + 4](sdf))
        sdf = torch.tanh(self.sdfLayers[7](sdf))

        if geometryOnly:
            return sdf

        # Texture portion
        texture = r
        for i in range(3):
            texture = self.activation(self.textureLayers[i](texture))
        texture = torch.sin(self.textureLayers[3](texture)) / 2.0 + 0.5
        return sdf, texture
