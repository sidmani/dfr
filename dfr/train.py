import torch
from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator
from .dataset import makeDataloader

def train(batchSize, device, dataset, steps, ckpt):
    hparams = ckpt.hparams
    print(hparams)

    dataloader = makeDataloader(batchSize, dataset, device)
    for idx in tqdm(range(ckpt.startEpoch, steps),
                    initial=ckpt.startEpoch,
                    total=steps):
        batch = next(dataloader)
        generated, normals = ckpt.gen.sample_like(batch)
        logData = {'fake': generated, 'real': batch}

        # update the generator every nth iteration
        if idx % hparams.discIter == 0:
            genData = stepGenerator(generated,
                                    normals,
                                    ckpt.dis,
                                    ckpt.genOpt,
                                    hparams.eikonalFactor)
            logData.update(genData)

        # update the discriminator
        disData = stepDiscriminator(generated, batch, ckpt.dis, ckpt.disOpt)
        logData.update(disData)

        # write the log output
        ckpt.log(logData, idx)

        # save every 25 iterations
        if idx % 25 == 0:
            ckpt.save(idx)
