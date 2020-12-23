import torch
import torch.nn as nn

# This discriminator is based on DC-GAN.
# - Weight init is not necessary and hasn't been shown to improve performance
# - Decreasing # of feature maps from 64 to 32 degrades results from FID=121 to FID=169 at epoch 70k.
# - Increasing # of feature maps from 64 to 128 improves from FID=121 to FID=96

# Possible improvements:
# - CoordConv
# - Various kinds of regularization (spectral norm; instance norm -> causes vanishing gradients)
# - Pooling across channels
# - Better architecture (including progressive growing)

# Questions:
# - Is the discriminator strong enough to handle multiple views of the object?

class ProgressiveBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation):
        super().__init__()
        # inChannels x S x S -> outChannels x S/2 x S/2
        self.layers = nn.Sequential(
            # pi-GAN uses in->out, out-> out, but pro-gan uses in->in, in->out
            nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            activation,
            nn.AvgPool2d(2),
        )
        self.inChannels = inChannels
        self.outChannels = outChannels

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, hparams, channels=4):
        super().__init__()
        self.channels = channels
        self.activation = nn.LeakyReLU(0.2)

        self.adapter = nn.ModuleList([])
        self.blocks = nn.ModuleList([
            # the last 2 blocks do 8x8 -> 2x2
            ProgressiveBlock(400, 400, self.activation),
            ProgressiveBlock(400, 400, self.activation)
        ])

        # 400x2x2 -> 1x1x1
        self.output = nn.Conv2d(400, 1, kernel_size=2)
        self.downsample = nn.AvgPool2d(2)

        # set up the progressive growing stages
        for stage in hparams.trainingStages:
            conv = nn.Conv2d(self.channels, stage.discChannels, kernel_size=1)
            self.adapter.append(conv)

            outChannels = self.blocks[0].inChannels
            block = ProgressiveBlock(stage.discChannels, outChannels, self.activation)
            self.blocks.insert(0, block)

        self.alpha = 1.

    def setStage(self, idx):
        self.stage = idx

    def setAlpha(self, alpha):
        self.alpha = alpha

    def forward(self, img):
        # the block corresponding to the current stage
        x = self.adapter[self.stage](img)
        x = self.blocks[-3 - self.stage](x)

        # the faded value from the previous stage
        if self.alpha < 1.0:
            x2 = self.downsample(img)
            x2 = self.adapter[self.stage - 1](x2)
            x = (1 - self.alpha) * x2 + self.alpha * x
            del x2

        for block in self.blocks[-3 - self.stage + 1:]:
            x = block(x)

        return self.output(x)
