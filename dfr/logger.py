import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, version, gradientData=False):
        self.logger = SummaryWriter(log_dir=f'runs/v{version}')
        self.gradientData = gradientData

    def write(self, data, idx):
        if 'generator_loss' in data:
            self.logger.add_scalar('generator/total', data['generator_loss'], global_step=idx)
            self.logger.add_scalar('generator/eikonal', data['eikonal_loss'], global_step=idx)

        self.logger.add_scalar('discriminator/fake', data['discriminator_fake'], global_step=idx)
        self.logger.add_scalar('discriminator/real', data['discriminator_real'], global_step=idx)
        self.logger.add_scalar('discriminator/total', data['discriminator_total'], global_step=idx)

        # log images every 10 iterations
        if idx % 10 == 0:
            fake = data['fake']
            real = data['real']
            self.logger.add_images('fake/collage', fake[:, :3], global_step=idx)
            self.logger.add_image('fake/closeup', fake[0][:3], global_step=idx)
            self.logger.add_image('fake/silhouette', fake[0][3], dataformats='HW', global_step=idx)
            self.logger.add_image('real/real', real[0][:3], global_step=idx)
            self.logger.add_image('real/silhouette', real[0][3], dataformats='HW', global_step=idx)

        # log debug data about discriminator gradients as necessary
        if self.gradientData and idx % 30 == 0:
            self.debug_gradientData(data, idx)

    def debug_gradientData(self, data, idx):
        fake = data['fake']
        real = data['real']
        gen, dis = data['models']
        real.requires_grad = True

        fakeOutput = dis(fake)
        realOutput = dis(real)

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
