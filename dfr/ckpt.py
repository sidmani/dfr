import re
import torch
from torch.optim import Adam
from .discriminator import Discriminator
from .sdfNetwork import SDFNetwork
from .hparams import HParams
from .flags import Flags
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
  def __init__(self, runDir, version, device, epoch=None, fork=None, override=False):
    # if no version is provided, create one
    if version is None:
      version = nextVersion(runDir)

    # get the latest epoch for the provided version
    if fork is None:
      self.loc = runDir / version
    else:
      self.loc = runDir / fork
      if self.loc.exists():
        raise Exception(f'Version {fork} already exists!')

    if epoch is None:
      epoch = latestEpoch(runDir / version)

    # if the version exists, load it
    if epoch is not None:
      print(f'Loading epoch {epoch}.')
      ckpt = torch.load(runDir / version / f"e{epoch}.pt", map_location=device)
      if override:
        self.hparams = HParams()
        print('Warning: overriding checkpoint hparams. Proceed at your own risk...')
      else:
        self.hparams = ckpt['hparams']
      # self.examples = ckpt['examples']
      self.startEpoch = epoch + 1
    else:
      ckpt = None
      self.hparams = HParams()
      # self.examples = torch.normal(
      #         mean=0.0,
      #         std=1.0,
      #         size=(3, self.hparams.latentSize),
      #         device=device)
      self.startEpoch = 0

    self.gen = SDFNetwork(self.hparams).to(device)
    self.dis = Discriminator(self.hparams, resolution=self.hparams.imageSize, channels=4).to(device)
    self.gradScaler = GradScaler(init_scale=2048., enabled=Flags.AMP)
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
    self.loc.mkdir(exist_ok=True)
    if overwrite:
      saved = self.loc.glob('*.pt')
      for file in saved:
        match = re.match("e([0-9]+)", str(file.stem))
        fileIdx = int(match[1])
        if fileIdx % 100 == 0 and epoch - fileIdx < 1000:
          continue
        if fileIdx  % 1000 == 0 and epoch - fileIdx < 10000:
          continue
        if fileIdx % 10000 == 0 and fileIdx != 0:
          continue

        file.unlink()

    torch.save({
      'hparams': self.hparams,
      'gen': self.gen.state_dict(),
      'dis': self.dis.state_dict(),
      'gen_opt': self.genOpt.state_dict(),
      'dis_opt': self.disOpt.state_dict(),
      'gradScaler': self.gradScaler.state_dict(),
      # 'examples': self.examples,
      }, self.loc / f"e{epoch}.pt")
