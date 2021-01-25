from .raycast import sample
from tqdm import tqdm
from .optim import stepDiscriminator, stepGenerator
from .flags import Flags

def train(dataloader, steps, ckpt, logger):
    for epoch in tqdm(ckpt.startEpoch, steps, initial=ckpt.startEpoch, total=steps):
        loop(dataloader, ckpt, logger, epoch)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, ckpt, logger, epoch):
    dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
    hparams = ckpt.hparams

    real = next(dataloader)
    # real = real[:, 3, :, :].unsqueeze(1)
    real.requires_grad = True

    # sample the generator for fake images
    batch = real.shape[0]
    sampled = sample(batch, real.device, ckpt, hparams.raycast, sigma, wide=stageIdx == 0)
    fake = sampled['full']
    # fake = sampled['full'][:, 3, :, :].unsqueeze(1)
    logData = {'fake': fake, 'real': real}

    disData = stepDiscriminator(real, fake, dis, disOpt, gradScaler, hparams.r1Factor)
    logData.update(disData)

    genData = stepGenerator(sampled, dis, genOpt, gradScaler, hparams.eikonal)
    logData.update(genData)

    gradScaler.update()

    if logger is not None:
        logger.log(logData, epoch)

    if not Flags.silent and epoch % 100 == 0:
        ckpt.save(epoch, overwrite=True)
