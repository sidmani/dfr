import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator
from .dataset import makeDataloader, ImageDataset
from .checkpoint import saveModel, loadModel

def train(batchSize, device, dataPath, dataCount, steps, version, checkpoint=None):
    models, optimizers, hparams, startEpoch = loadModel(checkpoint, device)
    gen, dis = models
    genOpt, disOpt = optimizers
    print(hparams)

    dataset = ImageDataset(dataPath, firstN=dataCount, imageSize=hparams.imageSize)
    dataloader = makeDataloader(batchSize, dataset, device)
    print(f"Starting at epoch {startEpoch}.")

    logger = SummaryWriter(log_dir=f'runs/v{version}')

    for idx in tqdm(range(startEpoch, steps), initial=startEpoch, total=steps):
        batch = next(dataloader)
        # sample the generator
        # don't use hparams.batchSize because the real batch may be smaller
        generated = gen.sample(batch.shape[0], device=device)

        # update the generator every nth iteration
        if idx % hparams.discIter == 0:
            stepGenerator(generated, dis, genOpt)

        # update the discriminator
        disReal, disFake = stepDiscriminator(generated, batch, dis, disOpt)

        # log loss every iteration
        logger.add_scalar('discriminator/fake', disFake, global_step=idx)
        logger.add_scalar('discriminator/real', disReal, global_step=idx)

        # save every 10 iterations, except idx 0
        if idx % 10 == 0 and idx != startEpoch:
            saveModel(gen, dis, genOpt, disOpt, hparams, version=version, epoch=idx, overwrite=True)
            logger.add_image('fake', generated[0], global_step=idx)
            logger.add_image('real', batch[0], global_step=idx)
