import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from antialiased_cnns import BlurPool
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

# Questions:
# - Why does loss jump when increasing scale? Shouldn't it be smooth?

# class EqualLR:
#     def __init__(self, name):
#         self.name = name

#     def compute_weight(self, module):
#         weight = getattr(module, self.name + '_orig')
#         fan_in = weight.data.size(1) * weight.data[0][0].numel()

#         return weight * np.sqrt(2 / fan_in)

#     @staticmethod
#     def apply(module, name):
#         fn = EqualLR(name)

#         weight = getattr(module, name)
#         del module._parameters[name]
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         module.register_forward_pre_hook(fn)

#         return fn

#     def __call__(self, module, input):
#         weight = self.compute_weight(module)
#         setattr(module, self.name, weight)

# def equal_lr(module, name='weight'):
#     EqualLR.apply(module, name)

#     return module

# class EqualConv2d(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#         conv = nn.Conv2d(*args, **kwargs)
#         conv.weight.data.normal_()
#         conv.bias.data.zero_()
#         self.conv = equal_lr(conv)

#     def forward(self, input):
#         return self.conv(input)

class ProgressiveBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation):
        super().__init__()
        # inChannels x S x S -> outChannels x S/2 x S/2
        self.layers = nn.Sequential(
            # pi-GAN uses in->out, out->out, but pro-gan uses in->in, in->out
            nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1),
            # EqualConv2d(inChannels, inChannels, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            # EqualConv2d(inChannels, outChannels, kernel_size=3, padding=1),
            activation,
            # aliasing may be a mild issue with the downsampled generator image
            # https://richzhang.github.io/antialiased-cnns/
            nn.AvgPool2d(2),
        )
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.ok = True

    def forward(self, x):
        assert self.ok
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
        self.stageCount = len(hparams.stages)

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
        self.hparams = hparams

    def setStage(self, idx):
        self.stage = idx

    def setAlpha(self, alpha):
        self.alpha = alpha

    def forward(self, img, sample='bilinear'):
        size = self.hparams.stages[self.stage].imageSize
        if img.shape[2] != size:
            full = torch.nn.functional.interpolate(img, size=(size, size), mode=sample)
        else:
            full = img

        # the block corresponding to the current stage
        x = self.adapter[self.stage](full)
        x = self.activation(x)
        x = self.blocks[self.stageCount - self.stage - 1](x)

        # the faded value from the previous stage
        if self.alpha < 1.0:
            # create the half-size image by directly downsampling from the original
            oldSize = self.hparams.stages[self.stage - 1].imageSize
            half = torch.nn.functional.interpolate(img, size=(oldSize, oldSize), mode=sample)
            x2 = self.adapter[self.stage - 1](half)
            x2 = self.activation(x2)
            # linear interpolation between new & old
            x = (1.0 - self.alpha) * x2 + self.alpha * x

        for block in self.blocks[self.stageCount - self.stage:]:
            x = block(x)

        # no sigmoid, since that's done by BCEWithLogitsLoss
        return self.output(x)
