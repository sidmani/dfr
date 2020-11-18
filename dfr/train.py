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
        generated, normals = gen.sample(batch.shape[0], device=device)

        # update the generator every nth iteration
        if idx % hparams.discIter == 0:
            genLoss, eikonalLoss = stepGenerator(generated, normals, dis, genOpt, hparams.eikonalFactor)
            logger.add_scalar('generator/total', genLoss, global_step=idx)
            logger.add_scalar('generator/eikonal', eikonalLoss, global_step=idx)

        # update the discriminator
        disReal, disFake, disTotal = stepDiscriminator(generated, batch, dis, disOpt)

        # log loss every iteration
        logger.add_scalar('discriminator/fake', disFake, global_step=idx)
        logger.add_scalar('discriminator/real', disReal, global_step=idx)
        logger.add_scalar('discriminator/total', disTotal, global_step=idx)

        # log every 10 iterations
        if idx % 10 == 0:
            logger.add_images('fake/collage', generated, global_step=idx)
            logger.add_image('fake/closeup', generated[0], global_step=idx)
            logger.add_image('real', batch[0], global_step=idx)

        # save every 25 iterations
        if idx % 25 == 0:
            saveModel(gen, dis, genOpt, disOpt, hparams, version=version, epoch=idx, overwrite=True)
