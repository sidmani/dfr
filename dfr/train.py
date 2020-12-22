import torch
import numpy as np
from .raycast import raycast
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator

def train(dataloader, steps, ckpt, logger):
    for idx in tqdm(range(ckpt.startEpoch, steps),
                    initial=ckpt.startEpoch,
                    total=steps):
        loop(dataloader, ckpt, logger, idx)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, ckpt, logger, idx):
    batch = next(dataloader)

    # sample the generator
    device = batch.device
    hparams = ckpt.hparams

    batchSize = batch.shape[0]
    phis = torch.ones(batchSize, device=device) * (np.pi / 6.0)
    thetas = torch.rand_like(phis) * (2.0 * np.pi)
    z = torch.normal(0.0,
                     hparams.latentStd,
                     size=(batchSize, hparams.latentSize),
                     device=device)
    sampled = raycast(phis, thetas, hparams.raycastSteps, hparams.fov, z, ckpt.gen, ckpt.gradScaler)

    fake = sampled['image']
    logData = {'fake': fake, 'real': batch}

    # update the generator every nth iteration
    if idx % hparams.discIter == 0:
        genData = stepGenerator(fake,
                                sampled['normals'],
                                sampled['illum'],
                                ckpt.dis,
                                ckpt.genOpt,
                                hparams.eikonalFactor,
                                hparams.illumFactor,
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
        ckpt.save(idx, overwrite=True)
