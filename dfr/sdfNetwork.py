import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input, gamma, beta):
        return torch.sin(self.omega_0 * (self.linear(input) * gamma + beta))

    def forward_debug(self, input, gamma, beta):
        z = self.omega_0 * (gamma * self.linear(input) + beta)
        return z, torch.sin(z)

class SDFNetwork(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        width = hparams.sdfWidth
        filmWidth = hparams.sdfWidth

        filmActivation = nn.LeakyReLU(0.2)
        self.film = nn.Sequential(
            nn.Linear(hparams.latentSize, filmWidth),
            filmActivation,
            nn.Linear(filmWidth, filmWidth),
            filmActivation,
            nn.Linear(filmWidth, width * 2),
        )

        self.layers = nn.ModuleList([
            SineLayer(3, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
        ])

        self.sdfLayers = nn.ModuleList([
            SineLayer(width, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
            nn.Linear(width, 1),
        ])

        self.txLayers = nn.ModuleList([
            SineLayer(width, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
            SineLayer(width, width, omega_0=hparams.sineOmega),
            nn.Linear(width, 3),
        ])

    def forward(self, pts, allLatents, mask, geomOnly=False):
        gamma, beta = torch.split(self.film(allLatents[mask]), self.hparams.sdfWidth, dim=1)

        r = pts
        for i in range(4):
            r = self.layers[i](r, gamma, beta)

        # sdf portion
        sdf = r
        if geomOnly:
            del r
        for i in range(3):
            sdf = self.sdfLayers[i](sdf, gamma, beta)
        sdf = self.sdfLayers[3](sdf)
        if geomOnly:
            return sdf

        # texture portion
        tx = r
        del r
        for i in range(3):
            tx = self.txLayers[i](tx, gamma, beta)
        tx = torch.sigmoid(self.txLayers[3](tx))

        return sdf, tx
