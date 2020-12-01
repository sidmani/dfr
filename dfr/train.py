from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator

def train(dataloader, steps, ckpt):
    for idx in tqdm(range(ckpt.startEpoch, steps),
                    initial=ckpt.startEpoch,
                    total=steps):
        batch = next(dataloader)
        generated, normals = ckpt.gen.sample_like(batch)
        logData = {'fake': generated, 'real': batch}

        # update the generator every nth iteration
        if idx % ckpt.hparams.discIter == 0:
            genData = stepGenerator(generated,
                                    normals,
                                    ckpt.dis,
                                    ckpt.genOpt,
                                    ckpt.hparams.eikonalFactor)
            logData.update(genData)

        # update the discriminator
        disData = stepDiscriminator(generated, batch, ckpt.dis, ckpt.disOpt)
        logData.update(disData)

        # write the log output
        ckpt.log(logData, idx)

        # save every 25 iterations
        if idx % 25 == 0:
            ckpt.save(idx)
