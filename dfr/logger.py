from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, version):
        self.logger = SummaryWriter(log_dir=f'runs/v{version}')

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

    def write_gradientData(self):
        pass
