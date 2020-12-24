import torch
import torch.nn as nn

# Progressive growing discriminator, based on pi-GAN architecture

# Possible improvements:
# - CoordConv
# - Minibatch std dev
# - Equalized learning rate
# - Skip connections
# - Various kinds of regularization (spectral norm; instance norm -> causes vanishing gradients)

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
            ProgressiveBlock(384, 384, self.activation),
            ProgressiveBlock(384, 384, self.activation)
        ])

        # 384x2x2 -> 1x1x1
        self.output = nn.Conv2d(384, 1, kernel_size=2)
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
