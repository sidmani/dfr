from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator

def train(dataloader, steps, ckpt, logger):
    for idx in tqdm(range(ckpt.startEpoch, steps),
                    initial=ckpt.startEpoch,
                    total=steps):
        batch = next(dataloader)
        generated, normals = ckpt.gen.sample_like(batch, ckpt.gradScaler)
        logData = {'fake': generated, 'real': batch}

        # update the generator every nth iteration
        if idx % ckpt.hparams.discIter == 0:
            genData = stepGenerator(generated,
                                    normals,
                                    ckpt.dis,
                                    ckpt.genOpt,
                                    ckpt.hparams.eikonalFactor,
                                    ckpt.gradScaler)
            logData.update(genData)

        # update the discriminator
        disData = stepDiscriminator(generated, batch, ckpt.dis, ckpt.disOpt, ckpt.gradScaler)
        logData.update(disData)

        # step the gradient scaler
        ckpt.gradScaler.update()

        if logger is not None:
            # write the log output
            logger.log(logData, idx)

        # save every 100 iterations
        if idx % 100 == 0:
            ckpt.save(idx)
