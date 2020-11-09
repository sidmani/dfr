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
                 imageSize=64):
        super().__init__()
        self.save_hyperparameters()

        self.gen = Generator(weightNorm=weightNorm,
                             fov=2.0 * np.pi / 3.0,
                             px=imageSize,
                             sampleCount=sampleCount,
                             latentSize=latentSize,
                             device=self.device)
        self.dis = Discriminator()

    def gradientPenalty(self, real, fake):
        epsilon = torch.rand(real.shape[0], 1, 1, device=self.device)
        interp = epsilon * real + (1.0 - epsilon) * fake
        outputs = self.dis(interp)
        grad = torch.autograd.grad(outputs=outputs,
                                   inputs=interp,
                                   grad_outputs=torch.ones(outputs.size(), device=self.device),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]

        return ((grad.norm(dim=1) - 1.0) ** 2.0).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        batchSize = batch.shape[0]
        genOpt, disOpt = self.optimizers()

        generated = self.gen.sample(batchSize)
        penalty = self.gradientPenalty(batch, generated)
        genLoss = self.dis(generated).mean()
        disLoss = (genLoss
                - self.dis(batch).mean()
                + penalty * self.hparams.gradPenaltyWeight)

        if batch_idx % self.hparams.discIter == 0:
            # reusing the generated data saves a generator pass each batch
            # graph: genLoss -> discriminator -> generator
            self.manual_backward(-genLoss, genOpt, retain_graph=True)
            # graph: disLoss -> genLoss -> discriminator -> generator
            # must run this backward pass before stepping the generator
            # otherwise the generator weights change and the graph becomes outdated
            self.manual_backward(disLoss, disOpt)
            genOpt.step()
            genOpt.zero_grad(set_to_none=True)
        else:
            self.manual_backward(disLoss, disOpt)

        disOpt.step()
        disOpt.zero_grad(set_to_none=True)

        # logging
        self.log('discriminator_loss', disLoss)
        self.log('generator_loss', genLoss)

        if batch_idx == 0:
            genImg = generated[torch.randint(generated.shape[0], (1,))]
            self.logger.experiment.add_image('generated_image',
                                      genImg,
                                      global_step=self.current_epoch,
                                      dataformats='CHW')

    def configure_optimizers(self):
        # TODO: custom beta value
        genOpt = Adam(self.gen.parameters(), self.hparams.learningRate)
        disOpt = Adam(self.dis.parameters(), self.hparams.learningRate)

        # TODO: learning rate schedule
        # schedGen = LambdaLR(optGen, lambda n: )
        return [genOpt, disOpt], []
