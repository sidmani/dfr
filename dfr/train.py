import torch
import numpy as np
from .raycast import sample_like
from .dataset import ImageDataset, makeDataloader
from tqdm import tqdm
from .optim import stepDiscriminator, stepGenerator
from .flags import Flags
from .image import blur, resample

def train(datapath, device, steps, ckpt, logger):
    stages = ckpt.hparams.stages
    dataset = ImageDataset(datapath)

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

        print(f'STAGE {i + 1}/{len(stages)}: resolution={stage.imageSize}, batch={stage.batch}.')
        dataloader = makeDataloader(stage.batch, dataset, device)
        for epoch in tqdm(range(startEpoch, endEpoch), initial=startEpoch, total=endEpoch):
            # fade in the new discriminator layer
            if stage.fade > 0:
                ckpt.dis.setAlpha(stage.evalAlpha(epoch))
            loop(dataloader, stages, i, ckpt, logger, epoch)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, stages, stageIdx, ckpt, logger, epoch):
    stage = stages[stageIdx]
    dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
    hparams = ckpt.hparams

    if stageIdx > 0:
        prevStage = stages[stageIdx - 1]
        sigma = dis.alpha * stage.sigma + (1 - dis.alpha) * prevStage.sigma
    else:
        sigma = stage.sigma

    with torch.no_grad():
        original = next(dataloader)
        blurred = blur(original, sigma)
        real_full = resample(blurred, stage.imageSize)
        real_full.requires_grad = True

        if dis.alpha < 1.:
            real_half = resample(blurred, size=prevStage.imageSize)
            real_half.requires_grad = True
        else:
            real_half = None

    # sample the generator for fake images
    sampled = sample_like(original, ckpt, stage.raycast, sigma / original.shape[2], half=dis.alpha < 1)
    fake = sampled['full']
    fakeHalf = sampled['half'] if 'half' in sampled else None
    logData = {'fake': fake, 'real': real_full, 'sigma': sigma}

    disData = stepDiscriminator(real_full, real_half, fake, fakeHalf, dis, disOpt, gradScaler, hparams.r1Factor)
    logData.update(disData)

    genData = stepGenerator(sampled, dis, genOpt, gradScaler, hparams.eikonal)
    logData.update(genData)

    # if 1.0 - logData['discriminator_real'].item() < 1e-4 and epoch > 2000:
    #     raise Exception('Training failed; discriminator is perfect.')

    # step the gradient scaler
    gradScaler.update()

    if logger is not None:
        # write the log output
        logger.log(logData, epoch)

    # save every 100 iterations
    if not Flags.silent and epoch % 100 == 0:
        ckpt.save(epoch, overwrite=True)
