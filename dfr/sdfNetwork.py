import torch
import torch.nn as nn
import numpy as np
from .positional import positional

class SDFNetwork(nn.Module):
    def __init__(self, hparams, width=512):
        super().__init__()
        inputSize = hparams.latentSize + 3 * hparams.positional * 2

        self.layers = nn.ModuleList([
            nn.Linear(inputSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - inputSize),
            # skip connection from input into 5th layer
            nn.Linear(width, width),
            nn.Linear(width, width),
        ])

        self.sdfLayers = nn.ModuleList([
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        self.txLayers = nn.ModuleList([
            nn.Linear(width, width),
            nn.Linear(width, 3),
        ])

        self.hparams = hparams
        if hparams.weightNorm:
            for i in range(len(self.layers)):
                self.layers[i] = nn.utils.weight_norm(self.layers[i])

            for i in range(len(self.sdfLayers)):
                self.sdfLayers[i] = nn.utils.weight_norm(self.sdfLayers[i])

            for i in range(len(self.txLayers)):
                self.txLayers[i] = nn.utils.weight_norm(self.txLayers[i])

        # DeepSDF uses ReLU, SALD uses Softplus
        self.activation = nn.ReLU()

    def forward(self, pts, expandedLatents, geomOnly=False):
        inp = torch.cat([positional(pts, self.hparams.positional), expandedLatents],
                         dim=1)
        r = inp
        # TODO: layer idx 1 can be split into separate matrices and cached
        for i in range(4):
            r = self.activation(self.layers[i](r))

        # skip connection
        r = torch.cat([inp, r], dim=1)
        for i in range(2):
            r = self.activation(self.layers[i + 4](r))

        # sdf portion
        sdf = self.activation(self.sdfLayers[0](r))
        sdf = torch.tanh(self.sdfLayers[1](sdf))

        if geomOnly:
            return sdf

        # texture portion
        tx = self.activation(self.txLayers[0](r))
        tx = torch.sigmoid(self.txLayers[1](tx))

        return sdf, tx
