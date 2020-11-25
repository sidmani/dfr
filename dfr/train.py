import torch
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator
from .dataset import makeDataloader, ImageDataset
from .checkpoint import saveModel, loadModel

def train(batchSize, device, dataPath, dataCount, steps, version, logger, checkpoint=None):
    models, optimizers, hparams, startEpoch = loadModel(checkpoint, device)
    gen, dis = models
    genOpt, disOpt = optimizers
    print(hparams)

    dataset = ImageDataset(dataPath, firstN=dataCount, imageSize=hparams.imageSize)
    dataloader = makeDataloader(batchSize, dataset, device)
    print(f"Starting at epoch {startEpoch}.")

    for idx in tqdm(range(startEpoch, steps), initial=startEpoch, total=steps):
        batch = next(dataloader)
        generated, normals = gen.sample_like(batch)
        logData = {'models': models, 'fake': generated, 'real': batch}

        # update the generator every nth iteration
        if idx % hparams.discIter == 0:
            genData = stepGenerator(generated, normals, dis, genOpt, hparams.eikonalFactor)
            logData.update(genData)

        # update the discriminator
        disData = stepDiscriminator(generated, batch, dis, disOpt)
        logData.update(disData)

        # write the log output
        logger.write(logData, idx)

        # save every 25 iterations
        if idx % 25 == 0:
            saveModel(gen, dis, genOpt, disOpt, hparams, version=version, epoch=idx, overwrite=True)
