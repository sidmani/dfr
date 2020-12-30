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
        self.logger.add_scalar('grad_scale', self.ckpt.gradScaler.get_scale(), global_step=idx)
        self.logger.add_scalar('discriminator/without_penalty', data['discriminator_fake'] - data['discriminator_real'], global_step=idx)

    def writeImages(self, data, idx):
        # log images every 50 iterations (every ~6 seconds on 64x64)
        fake = data['fake']
        real = data['real']
        self.logger.add_images('fake/collage', fake[:, :3], global_step=idx)
        self.logger.add_image('fake/closeup', fake[0][:3], global_step=idx)
        self.logger.add_image('fake/silhouette', fake[0][3], dataformats='HW', global_step=idx)
        self.logger.add_image('real/real', real[0][:3], global_step=idx)
        self.logger.add_image('real/silhouette', real[0][3], dataformats='HW', global_step=idx)

    def writeActivations(self, idx):
        pts = torch.randn(100, 3, device=self.ckpt.examples.device)
        latents = torch.normal(0.0,
                               self.ckpt.hparams.latentStd,
                               size=(100, self.ckpt.hparams.latentSize),
                               device=pts.device)
        z = []
        activations = []
        with torch.no_grad():
            gamma, beta = torch.split(self.ckpt.gen.film(latents), 512, dim=1)
            r = pts
            for i in range(4):
                lin, r = self.ckpt.gen.layers[i].forward_debug(r, gamma, beta)
                z.append(lin)
                activations.append(r)

            # sdf portion
            sdf = r
            for i in range(3):
                lin, sdf = self.ckpt.gen.sdfLayers[i].forward_debug(sdf, gamma, beta)
                z.append(lin)
                activations.append(sdf)

        self.logger.add_histogram(f'gamma', gamma, global_step=idx)
        self.logger.add_histogram(f'beta', beta, global_step=idx)

        for i in range(len(z)):
            self.logger.add_histogram(f'activation/{i}', activations[i], global_step=idx)
            self.logger.add_histogram(f'linear/{i}', z[i], global_step=idx)

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
