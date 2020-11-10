import torch
from torch.utils.data import DataLoader, get_worker_info
import numpy as np
from itertools import repeat
from tqdm import tqdm
import time
from collections import namedtuple
from torch.optim import Adam
from .discriminator import Discriminator
from .sdfNetwork import SDFNetwork
from .optim import stepGenerator, stepDiscriminator

torch.autograd.set_detect_anomaly(True)

HParams = namedtuple('HParams', [
        'learningRate',
        'raySamples',
        'weightNorm',
        'imageSize',
        'discIter',
        'latentSize',
        'fov',
        'batchSize',
    ], defaults=[
        1e-4,
        32,
        False,
        64,
        3,
        256,
        2.0 * np.pi / 3.0,
        4,
    ])

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader, device):
    for loader in repeat(dataloader):
        for data in loader:
            yield data.to(device)

def train(hparams, device, dataset, steps):
    dis = Discriminator()
    gen = SDFNetwork(hparams, device)
    # TODO: custom beta value
    # TODO: learning rate schedule
    genOpt = Adam(gen.parameters(), hparams.learningRate)
    disOpt = Adam(dis.parameters(), hparams.learningRate)

    print(hparams.batchSize)
    dataloader = iterData(DataLoader(dataset,
        batch_size=hparams.batchSize,
        pin_memory=True,
        shuffle=True,
        num_workers=1), device=device)

    for idx in tqdm(range(steps)):
        batch = next(dataloader)
        # gen, dis, genOpt, disOpt, batch, idx

        # sample the generator
        # don't use hparams.batchSize because the real batch may be smaller
        generated = gen.sampleGenerator(batch.shape[0])

        # update the generator every nth iteration
        if idx % hparams.discIter == 0:
            stepGenerator(generated, dis, genOpt)

        # update the discriminator
        stepDiscriminator(generated, batch, dis, disOpt)
