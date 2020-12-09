from tqdm import tqdm
from .optim import stepGenerator, stepDiscriminator
from torch.cuda.amp import GradScaler

def train(dataloader, steps, ckpt):
    gradScaler = GradScaler(init_scale=32768.)

    for idx in tqdm(range(ckpt.startEpoch, steps),
                    initial=ckpt.startEpoch,
                    total=steps):
        batch = next(dataloader)
        generated, normals = ckpt.gen.sample_like(batch, gradScaler)
        logData = {'fake': generated, 'real': batch}

        # update the generator every nth iteration
        if idx % ckpt.hparams.discIter == 0:
            genData = stepGenerator(generated,
                                    normals,
                                    ckpt.dis,
                                    ckpt.genOpt,
                                    ckpt.hparams.eikonalFactor,
                                    gradScaler)
            logData.update(genData)

        # update the discriminator
        disData = stepDiscriminator(generated, batch, ckpt.dis, ckpt.disOpt, gradScaler)
        logData.update(disData)

        # step the gradient scaler
        gradScaler.update()

        # write the log output
        ckpt.log(logData, idx)

        # save every 100 iterations
        if idx % 100 == 0:
            ckpt.save(idx)
