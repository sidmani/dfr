import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .raycast import sample_like
from .dataset import ImageDataset, makeDataloader
from tqdm import tqdm
from .optim import R1
from tools.grad_graph import register_hooks
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

        # stop execution if necessary
        if endEpoch <= startEpoch:
            break

        print(f'STAGE {i + 1}/{len(stages)}: resolution={stage.imageSize}, batch={stage.batch}.')
        dataloader = makeDataloader(stage.batch, dataset, device)

        for idx in tqdm(range(startEpoch, endEpoch), initial=startEpoch, total=endEpoch):
            # fade in the new discriminator layer
            if stage.fade > 0:
                ckpt.dis.setAlpha(min(1.0, float(idx - stage.start) / float(stage.fade)))
            loop(dataloader, stage, ckpt, logger, idx)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, stage, ckpt, logger, idx):
    # get the next batch of real images
    real = next(dataloader)

    # sample the generator for fake images
    sampled = sample_like(real, ckpt, stage.raycast, stage.sharpness)
    fake = sampled['image']
    logData = {'fake': fake, 'real': real}

    dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
    hparams = ckpt.hparams

    ### discriminator update ###
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    detachedFake = fake.detach().clone().requires_grad_()
    criterion = nn.BCEWithLogitsLoss()

    disOpt.zero_grad(set_to_none=True)
    real.requires_grad = True
    with autocast(enabled=Flags.AMP):
        disReal = dis(real).view(-1)
        label = torch.full((real.shape[0],), 1.0, device=disReal.device)
        disLossReal = criterion(disReal, label)

        label = torch.full((real.shape[0],), 0.0, device=disReal.device)
        disFake = dis(detachedFake).view(-1)
        disLossFake = criterion(disFake, label)

    # note that we need to apply sigmoid, since BCEWithLogitsLoss does that internally
    penalty = 10 * R1(real, torch.sigmoid(disReal), gradScaler)

    gradScaler.scale(disLossReal).backward(retain_graph=True)
    gradScaler.scale(disLossFake).backward()
    gradScaler.scale(penalty).backward()
    gradScaler.step(disOpt)

    logData['discriminator_real'] = torch.sigmoid(disReal).mean().detach()
    logData['discriminator_fake'] = torch.sigmoid(disFake).mean().detach()
    logData['penalty'] = penalty.detach()
    del disReal
    del disFake

    if 1.0 - logData['discriminator_real'].item() < 1e-4 and idx > 1000:
        raise Exception('Training failed; discriminator is perfect.')

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
        output = dis(fake).view(-1)
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
