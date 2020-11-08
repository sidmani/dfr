import torch
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
from .discriminator import Discriminator
from .generator import Generator

class GAN(pl.LightningModule):
    def __init__(self,
                 weightNorm=False,
                 learningRate=1e-4,
                 sampleCount=32,
                 latentSize=256,
                 # WGAN-gp lambda parameter
                 gradPenaltyWeight=10.0,
                 # discriminator iterations for every generator iteration
                 discIter=3,
                 px=64):
        super().__init__()
        self.save_hyperparameters()

        self.gen = Generator(weightNorm=weightNorm,
                             fov=2.0 * np.pi / 3.0,
                             px=px,
                             sampleCount=sampleCount,
                             latentSize=latentSize)
        self.dis = Discriminator()

    def sample_generator(self, batchSize):
        # elevation angle: phi = pi/6
        phis = torch.ones(batchSize, device=self.device) * (np.pi / 6.0)
        # azimuthal angle: 0 <= theta < 2pi
        thetas = torch.rand(batchSize, device=self.device) * (2.0 * np.pi)
        # latents with mean 0, variance 0.33
        z = torch.normal(
                mean=0.0,
                std=np.sqrt(0.33),
                size=(batchSize, self.hparams.latentSize),
                device=self.device)

        return self.gen(z, phis, thetas)

    def gradient_penalty(self, real, fake):
        epsilon = torch.rand(real.shape[0])
        interp = epsilon * real + (1.0 - epsilon) * fake
        outputs = self.discriminator(interp)
        grad = torch.autograd.grad(outputs=outputs,
                                   inputs=interp,
                                   grad_outputs=torch.ones(outputs.size(), device=self.device),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]

        return ((grad.norm(dim=1) - 1.0) ** 2.0).mean()

    def training_step(self, batch, batchIdx):
        batchSize = batch.shape[0]
        genOpt, disOpt = self.optimizers()

        # update the discriminator more frequently than the generator
        for i in range(self.hparams.discIter):
            generated = self.sampleGenerator(batchSize)
            penalty = self.gradient_penalty(batch, generated)
            disLoss = (self.dis(generated).mean()
                    - self.dis(batch).mean()
                    + penalty * self.hparams.gradPenaltyWeight)
            self.manual_backward(disLoss, disOpt)
            disOpt.step()
            disOpt.zero_grad()

        generated = self.sampleGenerator(batchSize)
        genLoss = -self.dis(generated).mean()
        self.manual_backward(genLoss, genOpt)
        genOpt.step()
        genOpt.zero_grad()

        self.log('discriminator_loss', disLoss)
        self.log('generator_loss', genLoss)

    def configure_optimizers(self):
        # TODO: custom beta value
        genOpt = Adam(self.gen.parameter(), self.hparams.learningRate)
        disOpt = Adam(self.dis.parameters(), self.hparams.learningRate)

        # TODO: learning rate
        # schedGen = LambdaLR(optGen, lambda n: )
        return [genOpt, disOpt], []
