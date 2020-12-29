import torch
import torch.nn as nn
from antialiased_cnns import BlurPool

# Progressive growing discriminator, based on pi-GAN architecture
# Possible improvements:
# - CoordConv
# - Minibatch std dev
# - Equalized learning rate
# - Skip connections
# - Various kinds of regularization; have tried the following:
#   - spectral norm causes mode collapse
#   - instance norm causes vanishing gradients

# Questions:
# - Why does loss jump when increasing scale? Shouldn't it be smooth?

class ProgressiveBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation):
        super().__init__()
        # inChannels x S x S -> outChannels x S/2 x S/2
        self.layers = nn.Sequential(
            # pi-GAN uses in->out, out->out, but pro-gan uses in->in, in->out
            nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            activation,
            # average pooling can cause problems
            # https://richzhang.github.io/antialiased-cnns/
            nn.AvgPool2d(2),
        )
        self.inChannels = inChannels
        self.outChannels = outChannels

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, hparams, channels=4):
        super().__init__()
        self.inChannels = channels
        self.activation = nn.LeakyReLU(0.2)

        self.adapter = nn.ModuleList([])
        self.blocks = nn.ModuleList([
            # the last 2 blocks do 8x8 -> 2x2
            ProgressiveBlock(384, 384, self.activation),
            ProgressiveBlock(384, 384, self.activation),
        ])
        self.initBlockCount = len(self.blocks)

        # 384x2x2 -> 1x1x1
        # even-sized convolutions have issues
        # https://papers.nips.cc/paper/2019/hash/2afe4567e1bf64d32a5527244d104cea-Abstract.html
        self.output = nn.Conv2d(384, 1, kernel_size=2)

        # set up the progressive growing stages
        for stage in hparams.stages:
            conv = nn.Conv2d(self.inChannels, stage.discChannels, kernel_size=1)
            self.adapter.append(conv)

            outChannels = self.blocks[0].inChannels
            block = ProgressiveBlock(stage.discChannels, outChannels, self.activation)
            self.blocks.insert(0, block)

        self.alpha = 1.

    def setStage(self, idx):
        self.stage = idx

    def setAlpha(self, alpha):
        self.alpha = alpha

    def forward(self, img, downsampled):
        # the block corresponding to the current stage
        x = self.adapter[self.stage](img)
        x = self.activation(x)
        x = self.blocks[-(self.initBlockCount + 1) - self.stage](x)

        # the faded value from the previous stage
        # assumes downsampled is not None
        if self.alpha < 1.0:
            # first stage can't have alpha < 1, so self.stage - 1 >= 0
            x2 = self.adapter[self.stage - 1](downsampled)
            x2 = self.activation(x2)
            x = (1 - self.alpha) * x2 + self.alpha * x
            del x2

        for block in self.blocks[-(self.initBlockCount + 1) - self.stage + 1:]:
            x = block(x)

        return self.output(x)
