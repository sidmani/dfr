import torch
import torch.nn as nn
import pytorch_lightning as pl
from .discriminator import Discriminator
from .generator import Generator

class GAN(pl.LightningModule):
    def __init__(self):
        self.dis = Discriminator()
        self.gen = Generator()

    # sample the generator
    def forward(self, x):
        return self.gen(x)

    def training_step(self, x):
        pass
