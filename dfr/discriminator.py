import torch
import torch.nn as nn
import numpy as np

class ResBlock(nn.Module):
  def __init__(self, inChannels, outChannels, activation):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1),
      activation,
      nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
      activation,
      nn.AvgPool2d(2),
    )

    self.skip = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False),
    )

  def forward(self, x):
    return (self.layers(x) + self.skip(x)) / np.sqrt(2)

class Discriminator(nn.Module):
  def __init__(self, hparams, resolution, channels):
    super().__init__()
    activation = nn.LeakyReLU(0.2)
    blocks = []
    logRes = int(np.floor(np.log2(resolution)))
    for i in range(2, logRes + 1):
      inChannels = hparams.channels[int(2 ** i)]
      outChannels = hparams.channels[int(2 ** (i - 1))]
      blocks.insert(0, ResBlock(inChannels, outChannels, activation))

    self.layers = nn.Sequential(
      nn.Conv2d(channels, hparams.channels[resolution], kernel_size=1),
      activation,
      nn.Sequential(*blocks),
      nn.Conv2d(hparams.channels[2], 1, kernel_size=2),
    )

  def forward(self, x):
    return self.layers(x)
