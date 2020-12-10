import torch
import torch.nn as nn
import numpy as np
from .positional import positional

class SDFNetwork(nn.Module):
    def __init__(self, hparams, basis, width=512):
        super().__init__()
        inputSize = hparams.latentSize + hparams.positionalSize * 2

        self.layers = nn.ModuleList([
            nn.Linear(inputSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            # skip connection from input into 5th layer
        ])

        self.sdfLayers = nn.ModuleList([
            nn.Linear(width + inputSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        self.txLayers = nn.ModuleList([
            nn.Linear(width + inputSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 3),
        ])

        self.basis = basis

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

    # this must be run with no_grad
    def forward_inplace(self, pts, allLatents, mask):
        expandedLatents = allLatents[mask]
        sin, cos = positional(pts, self.basis, inplace=True)
        inp = torch.cat([sin, cos, expandedLatents], dim=1)
        del sin
        del cos
        del expandedLatents

        a = inp
        for i in range(4):
            a = self.activation(self.layers[i](a))

        # skip connection
        a = torch.cat([inp, a], dim=1)
        del inp

        # sdf portion
        for i in range(3):
            a = self.activation(self.sdfLayers[i](a))

        return self.sdfLayers[3](a)

    def forward(self, pts, expandedLatents):
        sin, cos = positional(pts, self.basis)

        # a single cat operation here saves a lot of memory
        inp = torch.cat([sin, cos, expandedLatents], dim=1)
        del sin, cos

        r = inp
        for i in range(4):
            r = self.activation(self.layers[i](r))

        # skip connection
        r = torch.cat([inp, r], dim=1)
        del inp

        # sdf portion
        sdf = r
        for i in range(3):
            sdf = self.activation(self.sdfLayers[i](sdf))
        sdf = self.sdfLayers[3](sdf)

        # texture portion
        tx = r
        del r
        for i in range(3):
            tx = self.activation(self.txLayers[i](tx))
        tx = torch.sigmoid(self.txLayers[3](tx))

        return sdf, tx
