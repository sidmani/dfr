import torch
import numpy as np
import torch.nn as nn

class TextureNetwork(nn.Module):
    def __init__(self, hparams, width=512):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(hparams.latentSize + 3, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (hparams.latentSize + 3)),
            # skip connection from input: latent + x (into 4th layer)
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 3),
        ])

        self.activation = nn.ReLU()

        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, 1e-1)

        if hparams.weightNorm:
            for i in range(8):
                self.layers[i] = nn.utils.weight_norm(self.layers[i])

    def forward(self, x):
        r = x
        for i in range(4):
            r = self.activation(self.layers[i](r))

        # skip connection
        r = torch.cat([x, r], dim=1)
        for i in range(3):
            r = self.activation(self.layers[i + 4](r))
        return torch.sigmoid(self.layers[7](r))
