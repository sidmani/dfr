import torch
import numpy as np
from .raycast import raycast
from .dataset import ImageDataset, makeDataloader
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator

def train(datapath, device, steps, ckpt, logger, profile=False):
    stages = ckpt.hparams.trainingStages
    for i in range(ckpt.startStage, len(stages)):
        stage = stages[i]
        ckpt.dis.setStage(i)

        # start at the stage start, unless the checkpoint is from mid-stage
        startEpoch = max(stage.start, ckpt.startEpoch)
        # end at the epoch before the next stage, if the next stage exists
        if len(stages) > i + 1:
            endEpoch = min(stages[i + 1].start, steps)
        else:
            endEpoch = steps

        # stop execution if necessary
        if endEpoch <= startEpoch:
            break

        imageSize = np.prod(stage.raycast)
        print(f'STAGE {i + 1}/{len(stages)}: resolution={imageSize}x{imageSize}, batch={stage.batch}.')

        dataset = ImageDataset(datapath, imageSize=imageSize)
        dataloader = makeDataloader(stage.batch, dataset, device, workers=0 if profile else 1)

        for idx in tqdm(range(startEpoch, endEpoch), initial=startEpoch, total=endEpoch):
            # fade in the new discriminator layer
            if stage.fade > 0:
                ckpt.dis.setAlpha(min(1.0, float(idx - startEpoch) / float(stage.fade)))

            loop(dataloader, stage, ckpt, logger, idx)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, stage, ckpt, logger, idx):
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
    sampled = raycast(phis, thetas, stage.raycast, hparams.fov, z, ckpt.gen, ckpt.gradScaler)

    fake = sampled['image']
    logData = {'fake': fake, 'real': batch}

    # update the generator every nth iteration
    if idx % hparams.discIter == 0:
        genData = stepGenerator(fake,
                                sampled['normalLength'],
                                ckpt.dis,
                                ckpt.genOpt,
                                hparams.eikonalFactor,
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
