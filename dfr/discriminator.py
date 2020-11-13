import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=3, fmapSize=64):
        super().__init__()
        # DC-GAN discriminator architecture
        # batch norm omitted per WGAN-GP
        # bias=False is only used with batch norm
        self.layers = nn.ModuleList([
            nn.Conv2d(channels, fmapSize, 4, 2, 1),
            nn.Conv2d(fmapSize, fmapSize * 2, 4, 2, 1),
            nn.Conv2d(fmapSize * 2, fmapSize * 4, 4, 2, 1),
            nn.Conv2d(fmapSize * 4, fmapSize * 8, 4, 2, 1),
            nn.Conv2d(fmapSize * 8, 1, 4, 1, 0)
        ])

        # weight init, according to DC-GAN
        for layer in self.layers:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)

        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        for i in range(4):
            x = self.activation(self.layers[i](x))
        return torch.sigmoid(self.layers[4](x)).squeeze()
