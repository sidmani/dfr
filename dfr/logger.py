import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .raycast import raycast

class Logger:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.logger = SummaryWriter(ckpt.loc)

    def log(self, data, idx):
        self.writeScalars(data, idx)

        if idx % 100 == 0:
            self.writeImages(data, idx)

        # if idx % 200 == 0:
        #     self.writeFixedSamples(idx)

    def writeScalars(self, data, idx):
        self.logger.add_scalar('generator/total', data['generator_loss'], global_step=idx)
        self.logger.add_scalar('generator/eikonal', data['eikonal_loss'], global_step=idx)

        self.logger.add_scalar('discriminator/fake', data['discriminator_fake'], global_step=idx)
        self.logger.add_scalar('discriminator/real', data['discriminator_real'], global_step=idx)
        self.logger.add_scalar('discriminator/total', data['discriminator_total'], global_step=idx)
        self.logger.add_scalar('discriminator/penalty', data['penalty'], global_step=idx)
        self.logger.add_scalar('discriminator/alpha', self.ckpt.dis.alpha, global_step=idx)

        self.logger.add_scalar('grad_scale', self.ckpt.gradScaler.get_scale(), global_step=idx)

    def writeImages(self, data, idx):
        fake = data['fake'].clamp(0, 1)
        real = data['real']
        self.logger.add_images('fake/collage', fake, dataformats='NCHW', global_step=idx)
        # self.logger.add_image('fake/closeup', fake[0][:3], global_step=idx)
        self.logger.add_image('fake/silhouette', fake[0].squeeze(0), dataformats='HW', global_step=idx)
        # self.logger.add_image('real/real', real[0][:3], global_step=idx)
        self.logger.add_image('real/silhouette', real[0].squeeze(0), dataformats='HW', global_step=idx)

    # def writeFixedSamples(self, idx):
    #     device = self.ckpt.examples.device
    #     phis = torch.ones(3, device=device) * np.pi / 6.
    #     thetas = torch.tensor([-0.25, 0.0, 0.25], device=device) * np.pi
    #     hp = self.ckpt.hparams
    #     result = raycast(phis,
    #                    thetas,
    #                    hp.raycastSteps,
    #                    hp.fov,
    #                    self.ckpt.examples,
    #                    self.ckpt.gen,
    #                    self.ckpt.gradScaler)
    #     imgs = result['image']
    #     self.logger.add_images('fake/fixed_sample', imgs[:, :3], global_step=idx)
