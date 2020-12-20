import re
import torch
from torch.optim import Adam
from .discriminator import Discriminator
from .sdfNetwork import SDFNetwork
from .raycast import MultiscaleFrustum
from .hparams import HParams
from torch.cuda.amp import GradScaler

def latestEpoch(loc):
    if not loc.exists:
        return None

    available = list(loc.glob('*.pt'))
    if len(available) == 0:
        return None

    nums = []
    for f in available:
        match = re.match("e([0-9]+)", str(f.stem))
        nums.append(int(match[1]))
    return max(nums)

def nextVersion(runDir):
    versions = [-1]
    for f in runDir.glob('*'):
        match = re.match('([0-9]+)$', str(f.stem))
        if match:
            versions.append(int(match[1]))

    return str(max(versions) + 1)

class Checkpoint:
    def __init__(self, runDir, version, device, epoch=None, noLog=False):
        self.noLog = noLog

        # if no version is provided, create one
        if version is None:
            version = nextVersion(runDir)

        # get the latest epoch for the provided version
        self.loc = runDir / version
        if epoch is None:
            epoch = latestEpoch(self.loc)

        # if the version exists, load it
        if epoch is not None:
            ckpt = torch.load(self.loc / f"e{epoch}.pt", map_location=device)
            self.hparams = ckpt['hparams']
            self.examples = ckpt['examples']
            self.startEpoch = epoch + 1

        else:
            ckpt = None
            if not noLog:
                self.loc.mkdir(exist_ok=True)
            self.hparams = HParams()
            self.examples = torch.normal(
                    mean=0.0,
                    std=self.hparams.latentStd,
                    size=(3, self.hparams.latentSize),
                    device=device)
            self.startEpoch = 0

        self.frustum = MultiscaleFrustum(self.hparams.fov, self.hparams.raycastSteps, device=device)
        self.gen = SDFNetwork(self.hparams).to(device)
        self.dis = Discriminator(self.hparams).to(device)
        self.gradScaler = GradScaler(init_scale=32768.)

        self.genOpt = Adam(self.gen.parameters(),
                           self.hparams.learningRate,
                           betas=self.hparams.betas)
        self.disOpt = Adam(self.dis.parameters(),
                           self.hparams.learningRate,
                           betas=self.hparams.betas)

        if ckpt is not None:
            self.dis.load_state_dict(ckpt['dis'])
            self.gen.load_state_dict(ckpt['gen'])
            self.genOpt.load_state_dict(ckpt['gen_opt'])
            self.disOpt.load_state_dict(ckpt['dis_opt'])
            self.gradScaler.load_state_dict(ckpt['gradScaler'])

    def save(self, epoch, overwrite=True):
        if self.noLog:
            return

        # TODO: FID top-5
        if overwrite:
            for file in self.loc.glob("*.pt"):
                file.unlink()

        torch.save({
            'hparams': self.hparams,
            'gen': self.gen.state_dict(),
            'dis': self.dis.state_dict(),
            'gen_opt': self.genOpt.state_dict(),
            'dis_opt': self.disOpt.state_dict(),
            'gradScaler': self.gradScaler.state_dict(),
            'examples': self.examples,
            }, self.loc / f"e{epoch}.pt")
