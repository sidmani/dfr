import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .raycast import sample_like
from .dataset import ImageDataset, makeDataloader
from tqdm import tqdm
from .optim import R1
from .flags import Flags

def train(datapath, device, steps, ckpt, logger):
    stages = ckpt.hparams.stages
    dataset = ImageDataset(datapath)

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

        print(f'STAGE {i + 1}/{len(stages)}: resolution={stage.imageSize}, batch={stage.batch}.')
        dataloader = makeDataloader(stage.batch, dataset, device)
        for idx in tqdm(range(startEpoch, endEpoch), initial=startEpoch, total=endEpoch):
            # fade in the new discriminator layer
            if stage.fade > 0:
                ckpt.dis.setAlpha(min(1.0, float(idx - stage.start) / float(stage.fade)))
                # ckpt.dis.setAlpha(0)
            loop(dataloader, stages, i, ckpt, logger, idx)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, stages, stageIdx, ckpt, logger, idx):
    stage = stages[stageIdx]

    # get the next batch of real images
    real = next(dataloader)
    dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
    hparams = ckpt.hparams

    # fade sharpness during transition
    # TODO: set halfSharpness to none when not fading
    if stageIdx < len(stages) - 1:
        nextStage = stages[stageIdx + 1]
        gamma = min(max(0.,  1. - (nextStage.start - idx) / hparams.sharpnessFadeIn), 1.)
        fullSharpness = stage.sharpness * (1 - gamma) + nextStage.sharpness * gamma
    else:
        gamma = 0.
        fullSharpness = stage.sharpness

    if stageIdx > 0:
        halfSharpness = fullSharpness
    else:
        halfSharpness = None

    # sample the generator for fake images
    sampled = sample_like(real, ckpt, stage.raycast, fullSharpness, halfSharpness)
    fake = sampled['full']
    fakeHalf = sampled['half'] if stageIdx > 0 else None
    ds_real = torch.nn.functional.interpolate(real, size=(stage.imageSize, stage.imageSize), mode='bilinear')
    logData = {'fake': fake, 'real': ds_real, 'full_sharpness':fullSharpness, 'gamma': gamma}

    ### discriminator update ###
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    detachedFake = fake.detach().clone().requires_grad_()
    detachedFakeHalf = fakeHalf.detach().clone().requires_grad_() if stageIdx > 0 else None
    criterion = nn.BCEWithLogitsLoss()

    disOpt.zero_grad(set_to_none=True)
    real.requires_grad = True
    with autocast(enabled=Flags.AMP):
        disReal = dis(real).view(-1)
        label = torch.full((real.shape[0],), 1.0, device=disReal.device)
        disLossReal = criterion(disReal, label)

        label = torch.full((real.shape[0],), 0.0, device=disReal.device)
        disFake = dis(detachedFake, detachedFakeHalf).view(-1)
        disLossFake = criterion(disFake, label)

    # note that we need to apply sigmoid, since BCEWithLogitsLoss does that internally
    penalty = hparams.r1Factor * R1(real, torch.sigmoid(disReal), gradScaler)

    gradScaler.scale(disLossReal).backward(retain_graph=True)
    gradScaler.scale(disLossFake).backward()
    gradScaler.scale(penalty).backward()
    gradScaler.step(disOpt)

    with torch.no_grad():
        real_score = torch.sigmoid(disReal).mean().detach()
        fake_score = torch.sigmoid(disFake).mean().detach()
        logData['discriminator_real'] = real_score
        logData['discriminator_fake'] = fake_score
        logData['discriminator_total'] = fake_score - real_score

    logData['penalty'] = penalty.detach()
    del disReal
    del disFake

    # if 1.0 - logData['discriminator_real'].item() < 1e-4 and idx > 2000:
    #     raise Exception('Training failed; discriminator is perfect.')

    # disable autograd on discriminator params
    for p in dis.parameters():
        p.requires_grad = False

    genOpt.zero_grad(set_to_none=True)
    with autocast(enabled=Flags.AMP):
        # normals have already been scaled to correct values
        # the eikonal loss encourages the sdf to have unit gradient
        eikonalLoss = ((sampled['normalLength'] - 1.0) ** 2.0).mean()

        # the discriminator has been updated so we have to run the forward pass again
        # see https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/12
        label.fill_(1.)
        output = dis(fake, fakeHalf).view(-1)
        genLoss = criterion(output, label) + hparams.eikonal * eikonalLoss

    # graph: genLoss -> discriminator -> generator
    gradScaler.scale(genLoss).backward()
    gradScaler.step(genOpt)

    for p in dis.parameters():
        p.requires_grad = True

    logData['generator_loss'] = genLoss.detach()
    logData['eikonal_loss'] = eikonalLoss.detach()
    del genLoss
    del eikonalLoss

    # step the gradient scaler
    gradScaler.update()

    if logger is not None:
        # write the log output
        logger.log(logData, idx)

    # save every 100 iterations
    if not Flags.silent and idx % 100 == 0:
        ckpt.save(idx, overwrite=True)
