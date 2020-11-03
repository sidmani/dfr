import torch
import torch.nn as nn
import numpy as np
from .geometric import geometric_init

class DeepSDFDecoder(nn.Module):
    def __init__(self,
                 latentSize=256,
                 width=512,
                 weightNorm=False):

        assert width > latentSize + 3

        self.layers = nn.ModuleList([
            nn.Linear(latentSize + 3, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (latentSize + 3)),
            # skip connection from input: latent + x (into 5th layer)
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        # SAL geometric initialization
        geometric_init(self.layers)

        if weightNorm:
            for i in range(8):
                self.layers[i] = nn.utils.weight_norm(self.layers[i])

        self.activation = nn.ReLU

    def forward(self, x):
        r = x
        for i in range(7):
            if i == 4:
                # skip connection
                # per SAL supplementary: divide by sqrt(2) to normalize
                r = torch.cat([x, r], dim=1) / np.sqrt(2)
            r = self.activation(self.layers[i](r))

        # DeepSDF uses tanh, but SALD says it's optional
        return torch.tanh(self.layers[7](r))
