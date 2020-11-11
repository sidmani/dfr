import torch
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator
from .dataset import makeDataloader, ImageDataset
from .checkpoint import saveModel, loadModel

def train(batchSize, device, dataPath, dataCount, steps, version, checkpoint=None):
    dis, gen, disOpt, genOpt, hparams, startEpoch = loadModel(checkpoint, device)
    print(hparams)

    dataset = ImageDataset(dataPath, firstN=dataCount, imageSize=hparams.imageSize)
    dataloader = makeDataloader(batchSize, dataset, device)
    print(f"Starting at epoch {startEpoch}.")

    for idx in tqdm(range(startEpoch, steps)):
        batch = next(dataloader)
        # sample the generator
        # don't use hparams.batchSize because the real batch may be smaller
        generated = gen.sample(batch.shape[0], device=device)

        # update the generator every nth iteration
        if idx % hparams.discIter == 0:
            stepGenerator(generated, dis, genOpt)

        # update the discriminator
        stepDiscriminator(generated, batch, dis, disOpt)

        # save every n iterations, except idx 0
        if idx % 10 == 0 and idx != startEpoch:
            saveModel(gen, dis, genOpt, disOpt, hparams, version=version, epoch=idx, overwrite=True)
