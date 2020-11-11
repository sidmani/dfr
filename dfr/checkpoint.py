import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from collections import namedtuple
from .raycast.frustum import Frustum
from .discriminator import Discriminator
from .sdfNetwork import SDFNetwork
from .generator import Generator

HParams = namedtuple('HParams', [
        'learningRate',
        'raySamples',
        'weightNorm',
        'discIter',
        'latentSize',
        'fov',
        'imageSize',
    ], defaults=[
        1e-4,
        32,
        False,
        3,
        256,
        2.0 * np.pi / 3.0,
        64,
    ])

def saveModel(gen, dis, genOpt, disOpt, hparams, version, epoch, overwrite=True):
    ckptDir = Path.cwd() / 'runs' / f"v{version}"
    ckptDir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for file in ckptDir.glob("*.pt"):
            file.unlink()

    torch.save({
        'hparams': hparams,
        'gen': gen.state_dict(),
        'dis': dis.state_dict(),
        'gen_opt': genOpt.state_dict(),
        'dis_opt': disOpt.state_dict(),
        'epoch': epoch,
        }, ckptDir / f"e{epoch}.pt")

def loadModel(checkpoint, device):
    if checkpoint is not None:
        hparams = checkpoint['hparams']
        startEpoch = checkpoint['epoch'] + 1
    else:
        hparams = HParams()
        startEpoch = 0
    dis = Discriminator().to(device)

    # build generator
    sdf = SDFNetwork(hparams)
    frustum = Frustum(hparams.fov, hparams.imageSize, device)
    gen = Generator(sdf, frustum, hparams).to(device)
    models = (gen, dis)

    # TODO: custom beta value
    genOpt = Adam(gen.parameters(), hparams.learningRate)
    disOpt = Adam(dis.parameters(), hparams.learningRate)
    optimizers = (genOpt, disOpt)

    # TODO: learning rate schedule

    if checkpoint is not None:
        dis.load_state_dict(checkpoint['dis'])
        gen.load_state_dict(checkpoint['gen'])
        genOpt.load_state_dict(checkpoint['gen_opt'])
        disOpt.load_state_dict(checkpoint['dis_opt'])

    return models, optimizers, hparams, startEpoch
