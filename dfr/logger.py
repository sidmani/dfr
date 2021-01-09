import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .raycast import raycast

class Logger:
    def __init__(self, ckpt, gradientData=False, activations=False):
        self.gradientData = gradientData
        self.activations = activations
        self.ckpt = ckpt
        self.logger = SummaryWriter(ckpt.loc)

    def log(self, data, idx):
        self.writeScalars(data, idx)

        if idx % 50 == 0:
            self.writeImages(data, idx)
            if self.activations:
                self.writeActivations(idx)

        # if idx % 200 == 0:
        #     self.writeFixedSamples(idx)

        if idx % 10 == 0:
            self.writeGenScale(data, idx)

        if self.gradientData and idx % 30 == 0:
            self.writeGradientData(data, idx)

    def writeScalars(self, data, idx):
        if 'generator_loss' in data:
            self.logger.add_scalar('generator/total', data['generator_loss'], global_step=idx)
            self.logger.add_scalar('generator/eikonal', data['eikonal_loss'], global_step=idx)

        self.logger.add_scalar('discriminator/fake', data['discriminator_fake'], global_step=idx)
        self.logger.add_scalar('discriminator/real', data['discriminator_real'], global_step=idx)
        self.logger.add_scalar('discriminator/total', data['discriminator_total'], global_step=idx)
        self.logger.add_scalar('discriminator/penalty', data['penalty'], global_step=idx)
        self.logger.add_scalar('discriminator/alpha', self.ckpt.dis.alpha, global_step=idx)
        self.logger.add_scalar('grad_scale', self.ckpt.gradScaler.get_scale(), global_step=idx)

    def writeGenScale(self, data, idx):
        z = torch.normal(0.0, self.ckpt.hparams.latentStd, (12, self.ckpt.hparams.latentSize), device=self.ckpt.examples.device)
        film = self.ckpt.gen.film(z)
        split = torch.split(film, self.ckpt.hparams.sdfWidth, dim=1)
        norm = (split[0].norm(dim=1) + split[1].norm(dim=1)).mean().detach()
        self.logger.add_scalar('generator/film_scale', norm, global_step=idx)

    def writeImages(self, data, idx):
        # log images every 50 iterations (every ~6 seconds on 64x64)
        fake = data['fake']
        real = data['real']
        self.logger.add_images('fake/collage', fake[:, :3], global_step=idx)
        self.logger.add_image('fake/closeup', fake[0][:3], global_step=idx)
        self.logger.add_image('fake/silhouette', fake[0][3], dataformats='HW', global_step=idx)
        self.logger.add_image('real/real', real[0][:3], global_step=idx)
        self.logger.add_image('real/silhouette', real[0][3], dataformats='HW', global_step=idx)

    def writeFixedSamples(self, idx):
        device = self.ckpt.examples.device
        phis = torch.ones(3, device=device) * np.pi / 6.
        thetas = torch.tensor([-0.25, 0.0, 0.25], device=device) * np.pi
        hp = self.ckpt.hparams
        result = raycast(phis,
                       thetas,
                       hp.raycastSteps,
                       hp.fov,
                       self.ckpt.examples,
                       self.ckpt.gen,
                       self.ckpt.gradScaler)
        imgs = result['image']
        self.logger.add_images('fake/fixed_sample', imgs[:, :3], global_step=idx)

    def writeGradientData(self, data, idx):
        fake = data['fake']
        real = data['real']
        real.requires_grad = True

        fakeOutput = self.dis(fake)
        realOutput = self.dis(real)

        fakeGrad = torch.autograd.grad(outputs=fakeOutput,
                       inputs=fake,
                       grad_outputs=torch.ones_like(fakeOutput),
                       only_inputs=True)[0]
        realGrad = torch.autograd.grad(outputs=realOutput,
                       inputs=real,
                       grad_outputs=torch.ones_like(realOutput),
                       only_inputs=True)[0]

        fakeGrad = (fakeGrad ** 2.0).sum(dim=[1, 2, 3]).sqrt()
        realGrad = (realGrad ** 2.0).sum(dim=[1, 2, 3]).sqrt()

        self.logger.add_histogram('fake_gradient', fakeGrad, global_step=idx)
        self.logger.add_histogram('real_gradient', realGrad, global_step=idx)
