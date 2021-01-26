import torch
import torch.nn as nn
from .siren import SineLayer, siren_linear_init

# The SDF network is a SIREN, with FiLM conditioning, as in pi-GAN.
# - the FiLM network is run multiple times on the same latents; do forward pass before mask
# - the branches are deeper than in related architectures (like NeRF)

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
            SineLayer(3, width, omega=hparams.omega_first, is_first=True),
            SineLayer(width, width, omega=hparams.omega_hidden),
            SineLayer(width, width, omega=hparams.omega_hidden),
            SineLayer(width, width, omega=hparams.omega_hidden),
        ])

        self.sdfLayers = nn.ModuleList([
            SineLayer(width, width, omega=hparams.omega_hidden),
            SineLayer(width, width, omega=hparams.omega_hidden),
            SineLayer(width, width, omega=hparams.omega_hidden),
            nn.Linear(width, 1),
        ])

        self.txLayers = nn.ModuleList([
            SineLayer(width, width, omega=hparams.omega_hidden),
            SineLayer(width, width, omega=hparams.omega_hidden),
            SineLayer(width, width, omega=hparams.omega_hidden),
            nn.Linear(width, 3),
        ])

        siren_linear_init(self.sdfLayers[3], hparams.omega_hidden)
        siren_linear_init(self.txLayers[3], hparams.omega_hidden)

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
        tx = (torch.sin(self.txLayers[3](tx)) + 1) / 2

        return sdf, tx
