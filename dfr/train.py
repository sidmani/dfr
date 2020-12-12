import torch
from torch.cuda.amp import autocast
import numpy as np
from .raycast import raycast
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator

def sample_like(other, device, stdDev, phi):
    batch = other.shape[0]
    with torch.no_grad(), autocast:
        phis = torch.ones(batch, device=device) * phi
        thetas = torch.rand_like(phis) * (2.0 * np.pi)

def train(dataloader, steps, ckpt, logger, debugGenerator=False):
    for idx in tqdm(range(ckpt.startEpoch, steps),
                    initial=ckpt.startEpoch,
                    total=steps):
        batch = next(dataloader)

        # sample the generator
        device = batch.device
        batchSize = batch.shape[0]
        phis = torch.ones(batchSize, device=device) * (np.pi / 6.0)
        thetas = torch.rand_like(phis) * (2.0 * np.pi)
        z = torch.normal(0.0,
                         ckpt.hparams.latentStd,
                         size=(batchSize, ckpt.hparams.latentSize),
                         device=device)
        sampled = raycast(phis, thetas, ckpt.frustum, z, ckpt.gen, ckpt.gradScaler, debug=debugGenerator)

        fake = sampled['image']
        logData = {'fake': fake, 'real': batch}
        if debugGenerator:
            logData['normalMap'] = sampled['normalMap']
            logData['normalSizeMap'] = sampled['normalSizeMap']

        # update the generator every nth iteration
        if idx % ckpt.hparams.discIter == 0:
            genData = stepGenerator(fake,
                                    sampled['normals'],
                                    sampled['illum'],
                                    ckpt.dis,
                                    ckpt.genOpt,
                                    ckpt.hparams.eikonalFactor,
                                    ckpt.hparams.illumFactor,
                                    ckpt.gradScaler)
            logData.update(genData)

        # update the discriminator
        disData = stepDiscriminator(fake, batch, ckpt.dis, ckpt.disOpt, ckpt.gradScaler)
        logData.update(disData)

        # step the gradient scaler
        ckpt.gradScaler.update()

        if logger is not None:
            # write the log output
            logger.log(logData, idx)

        # save every 100 iterations
        if idx % 100 == 0:
            ckpt.save(idx)
