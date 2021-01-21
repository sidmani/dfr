import torch
from .raycast import sample
from tqdm import tqdm
from .optim import stepDiscriminator, stepGenerator
from .flags import Flags

def train(dataset, device, steps, ckpt, logger):
    stages = ckpt.hparams.stages

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
        for epoch in tqdm(range(startEpoch, endEpoch), initial=startEpoch, total=endEpoch):
            # fade in the new discriminator layer as necessary
            ckpt.dis.setAlpha(stage.evalAlpha(epoch))
            loop(dataset, device, stages, i, ckpt, logger, epoch)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataset, device, stages, stageIdx, ckpt, logger, epoch):
    stage = stages[stageIdx]
    dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
    hparams = ckpt.hparams

    with torch.no_grad():
        real = dataset.sample(stage.batch, res=stage.imageSize).to(device)
        real = real[:, 3, :, :].unsqueeze(1)
        real.requires_grad = True

    # sample the generator for fake images
    sampled = sample(stage.batch, device, ckpt, stage.raycast, stage.sigma)
    fake = sampled['full'][:, 3, :, :].unsqueeze(1)
    logData = {'fake': fake, 'real': real}

    disData = stepDiscriminator(real, fake, dis, disOpt, gradScaler, hparams.r1Factor)
    logData.update(disData)

    genData = stepGenerator(sampled, dis, genOpt, gradScaler, hparams.eikonal)
    logData.update(genData)

    # if 1.0 - logData['discriminator_real'].item() < 1e-4 and epoch > 2000:
    #     raise Exception('Training failed; discriminator is perfect.')

    # step the gradient scaler
    gradScaler.update()

    if logger is not None:
        logger.log(logData, epoch)

    if not Flags.silent and epoch % 100 == 0:
        ckpt.save(epoch, overwrite=True)
