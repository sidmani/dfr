import torch
import numpy as np
from torch.cuda.amp import autocast
from .raycast import sample_like
from .dataset import ImageDataset, makeDataloader
from tqdm import tqdm
from .optim import gradientPenalty

def train(datapath, device, steps, ckpt, logger, profile=False):
    stages = ckpt.hparams.stages
    dataset = ImageDataset(datapath, [s.imageSize for s in stages])

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

        # stop execution if necessary
        if endEpoch <= startEpoch:
            break

        print(f'STAGE {i + 1}/{len(stages)}: resolution={stage.imageSize}, batch={stage.batch}.')
        request = [stage.imageSize]
        if i > 0:
            request.append(stages[i - 1].imageSize)
        dataset.requestSizes(request)
        dataloader = makeDataloader(stage.batch, dataset, device, workers=0 if profile else 1)

        for idx in tqdm(range(startEpoch, endEpoch), initial=startEpoch, total=endEpoch):
            # fade in the new discriminator layer
            if stage.fade > 0:
                ckpt.dis.setAlpha(0)
                # ckpt.dis.setAlpha(min(1.0, float(idx - startEpoch) / float(stage.fade)))

            loop(dataloader, stage, ckpt, logger, idx)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, stage, ckpt, logger, idx):
    # get the next batch of real images
    batch = next(dataloader)
    realFull = batch[0]
    realHalf = batch[1] if len(batch) > 1 else None

    # sample the generator for fake images
    sampled = sample_like(realFull, ckpt, stage.raycast)
    fakeFull = sampled['image']
    fakeHalf = torch.nn.functional.avg_pool2d(fakeFull, 2) if len(batch) > 1 else None
    logData = {'fake': fakeFull, 'real': realFull}

    dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
    hparams = ckpt.hparams

    ### generator update ###
    if idx % hparams.discIter == 0:
        # disable autograd on disciminator params
        for p in dis.parameters():
            p.requires_grad = False

        with autocast():
            # normals have already been scaled to correct values
            # the eikonal loss encourages the sdf to have unit gradient
            eikonalLoss = ((sampled['normalLength'] - 1.0) ** 2.0).mean()

            # check what the discriminator thinks
            genLoss = -dis(fakeFull, fakeHalf).mean() + hparams.eikonal * eikonalLoss

        # graph: genLoss -> discriminator -> generator
        gradScaler.scale(genLoss).backward()
        gradScaler.step(genOpt)

        for p in dis.parameters():
            p.requires_grad = True

        # reset the generator, since it's done being differentiated
        genOpt.zero_grad(set_to_none=True)
        # the gen's params have changed, so can't backwards again on existing genLoss
        # so we have to run the discriminator again
        # see https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/12

        logData['generator_loss'] = genLoss.detach()
        logData['eikonal_loss'] = eikonalLoss.detach()
        del genLoss
        del eikonalLoss

    ### discriminator update ###
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    fakeFull = fakeFull.detach().clone().requires_grad_()
    if fakeHalf is not None:
        fakeHalf = fakeHalf.detach().clone().requires_grad_()

    penalty = gradientPenalty(dis, realFull, realHalf, fakeFull, fakeHalf, gradScaler)

    with autocast():
        disFake = dis(fakeFull, fakeHalf).mean()
        disReal = dis(realFull, realHalf).mean()
        disLoss = disFake - disReal + penalty * 10.0

    gradScaler.scale(disLoss).backward()
    gradScaler.step(disOpt)
    disOpt.zero_grad(set_to_none=True)

    # step the gradient scaler
    gradScaler.update()
    logData['discriminator_real'] = disReal.detach()
    logData['discriminator_fake'] = disFake.detach()
    logData['discriminator_total'] = disLoss.detach()

    if logger is not None:
        # write the log output
        logger.log(logData, idx)

    # save every 100 iterations
    if idx % 100 == 0:
        ckpt.save(idx, overwrite=True)
