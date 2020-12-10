import re
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from .discriminator import Discriminator
from .sdfNetwork import SDFNetwork
from .generator import Generator
from .raycast import MultiscaleFrustum
from .hparams import HParams
from .positional import createBasis

class Checkpoint:
    def __init__(self,
                 runDir,
                 version=None,
                 epoch=None,
                 device=None,
                 gradientData=False,
                 disableOutput=False):
        self.disableOutput = disableOutput

        if version:
            # load version if given
            self.loc = runDir / version
            if not self.loc.exists:
                raise Exception(f'Version {version} does not exist.')

            if not epoch:
                available = list(self.loc.glob('*.pt'))
                if len(available) == 0:
                    raise Exception(f'No checkpoints found for version {version}.')

                nums = []
                for f in available:
                    match = re.match("e([0-9]+)", str(f.stem))
                    nums.append(int(match[1]))
                epoch = max(nums)

            checkpoint = torch.load(self.loc / f"e{epoch}.pt", map_location=device)

            self.hparams = checkpoint['hparams']
            self.startEpoch = checkpoint['epoch'] + 1
            self.basis = checkpoint['basis']
        else:
            # otherwise create a new version
            versions = [-1]
            for f in runDir.glob('*'):
                match = re.match('([0-9]+)', str(f.stem))
                if match:
                    versions.append(int(match[1]))

            version = str(max(versions) + 1)
            self.loc = runDir / version

            if not disableOutput:
                self.loc.mkdir(exist_ok=True)

            checkpoint = None
            self.hparams = HParams()
            self.startEpoch = 0
            self.basis = createBasis(self.hparams.positionalSize, self.hparams.positionalScale, device)

        sdf = SDFNetwork(self.hparams, self.basis).to(device)
        frustum = MultiscaleFrustum(self.hparams.fov, self.hparams.raycastSteps, device=device)
        self.gen = Generator(sdf, frustum, self.hparams).to(device)
        self.dis = Discriminator(self.hparams).to(device)

        self.genOpt = Adam(self.gen.parameters(),
                           self.hparams.learningRate,
                           betas=self.hparams.betas)
        self.disOpt = Adam(self.dis.parameters(),
                           self.hparams.learningRate,
                           betas=self.hparams.betas)

        if checkpoint is not None:
            self.dis.load_state_dict(checkpoint['dis'])
            self.gen.load_state_dict(checkpoint['gen'])
            self.genOpt.load_state_dict(checkpoint['gen_opt'])
            self.disOpt.load_state_dict(checkpoint['dis_opt'])

        self.version = version

        if not disableOutput:
            self.logger = SummaryWriter(log_dir=self.loc)

        self.gradientData = gradientData

    def save(self, epoch, overwrite=True):
        if self.disableOutput:
            return

        if overwrite:
            for file in self.loc.glob("*.pt"):
                file.unlink()

        torch.save({
            'hparams': self.hparams,
            'gen': self.gen.state_dict(),
            'dis': self.dis.state_dict(),
            'gen_opt': self.genOpt.state_dict(),
            'dis_opt': self.disOpt.state_dict(),
            'basis': self.basis,
            'epoch': epoch,
            }, self.loc / f"e{epoch}.pt")

    def log(self, data, idx):
        if self.disableOutput:
            return

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
