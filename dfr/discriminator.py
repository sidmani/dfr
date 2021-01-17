import torch
import torch.nn as nn
import numpy as np
from tools.stats import tensor_stats

# Progressive growing discriminator, based on pi-GAN architecture
# Possible improvements:
# - CoordConv
# - Minibatch std dev
# - Equalized learning rate
# - Skip connections
# - Various kinds of regularization; have tried the following:
#   - spectral norm causes mode collapse
#   - instance norm causes vanishing gradients

class ProgressiveBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation):
        super().__init__()
        # inChannels x S x S -> outChannels x S/2 x S/2
        self.layers = nn.Sequential(
            # pi-GAN uses in->out, out->out, but pro-gan uses in->in, in->out
            # the first one uses fewer params
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(inChannels),
            activation,
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(outChannels),
            activation,
            nn.AvgPool2d(2),
        )
        # nn.init.constant_(self.layers[0].bias, 0)
        # nn.init.constant_(self.layers[2].bias, 0)
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.activation = activation

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, hparams, channels=4):
        super().__init__()
        self.inChannels = channels
        self.activation = nn.LeakyReLU(0.2)

        self.adapter = nn.ModuleList([])
        self.blocks = nn.ModuleList([ProgressiveBlock(384, 384, self.activation)])
        self.stageCount = len(hparams.stages)

        # 384x2x2 -> 1x1x1
        # this is effectively a linear layer
        self.output = nn.Conv2d(384, 1, kernel_size=2)

        # set up the progressive growing stages
        for idx, stage in enumerate(hparams.stages):
            conv = nn.Conv2d(self.inChannels, stage.discChannels, kernel_size=1)
            self.adapter.append(conv)

            outChannels = self.blocks[0].inChannels
            block = ProgressiveBlock(stage.discChannels, outChannels, self.activation)
            self.blocks.insert(0, block)

        self.alpha = 1.
        self.hparams = hparams

    def setStage(self, idx):
        self.stage = idx

    def setAlpha(self, alpha):
        self.alpha = alpha

    def forward(self, img):
        # the block corresponding to the current stage
        x = self.adapter[self.stage](img)
        x = self.activation(x)
        x = self.blocks[self.stageCount - self.stage - 1](x)

        # the faded value from the previous stage
        if self.alpha < 1.0:
            half = torch.nn.functional.avg_pool2d(img, 2)
            x2 = self.adapter[self.stage - 1](half)
            x2 = self.activation(x2)
            # linear interpolation between new & old
            x = (1.0 - self.alpha) * x2 + self.alpha * x

        for block in self.blocks[self.stageCount - self.stage:]:
            x = block(x)

        # no sigmoid, since that's done by BCEWithLogitsLoss
        return self.output(x)
