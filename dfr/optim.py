import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .flags import Flags

criterion = nn.BCEWithLogitsLoss()

# R1 gradient penalty (Mescheder et al., 2018)
def R1(real, realHalf, disReal, gradScaler):
    if realHalf is not None:
        inputs = (real, realHalf)
    else:
        inputs = real
    scaledGrad = torch.autograd.grad(outputs=gradScaler.scale(disReal),
                                     inputs=inputs,
                                     grad_outputs=torch.ones_like(disReal),
                                     create_graph=True,
                                     retain_graph=True,
                                     only_inputs=True)
    scale = gradScaler.get_scale()
    grad = [g / scale for g in scaledGrad]
    with autocast(enabled=Flags.AMP):
        # note that grad has shape NCHW
        # so we sum over channel, height, weight dims
        # and take mean over batch (N) dimension
        total = (grad[0] ** 2.).sum(dim=[1, 2, 3])
        if realHalf is not None:
            total = total + (grad[1] ** 2.).sum(dim=[1, 2, 3])
        return total.mean()

def stepDiscriminator(real, realHalf, fake, fakeHalf, dis, disOpt, gradScaler, r1Factor):
    ### discriminator update ###
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    detachedFake = fake.detach().clone().requires_grad_()
    detachedFakeHalf = fakeHalf.detach().clone().requires_grad_() if fakeHalf is not None else None

    disOpt.zero_grad(set_to_none=True)
    real.requires_grad = True
    with autocast(enabled=Flags.AMP):
        disReal = dis(real, realHalf).view(-1)
        label = torch.full((real.shape[0],), 1.0, device=disReal.device)
        disLossReal = criterion(disReal, label)

        label = torch.full((real.shape[0],), 0.0, device=disReal.device)
        disFake = dis(detachedFake, detachedFakeHalf).view(-1)
        disLossFake = criterion(disFake, label)

    # note that we need to apply sigmoid, since BCEWithLogitsLoss does that internally
    penalty = r1Factor * R1(real, realHalf, torch.sigmoid(disReal), gradScaler)

    gradScaler.scale(disLossReal + disLossFake + penalty).backward()
    gradScaler.step(disOpt)

    logData = {}
    with torch.no_grad():
        real_score = torch.sigmoid(disReal).mean().detach()
        fake_score = torch.sigmoid(disFake).mean().detach()
        logData['discriminator_real'] = real_score
        logData['discriminator_fake'] = fake_score
        logData['discriminator_total'] = fake_score - real_score
        logData['penalty'] = penalty.detach()

    return logData

def stepGenerator(sampled, dis, genOpt, gradScaler, eikonal):
    fake = sampled['full']
    fakeHalf = sampled['half'] if 'half' in sampled else None

    for p in dis.parameters():
        p.requires_grad = False

    genOpt.zero_grad(set_to_none=True)
    with autocast(enabled=Flags.AMP):
        # normals have already been scaled to correct values
        # the eikonal loss encourages the sdf to have unit gradient
        eikonalLoss = ((sampled['normalLength'] - 1.0) ** 2.0).mean()

        # the discriminator has been updated so we have to run the forward pass again
        # see https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/12
        label = torch.full((fake.shape[0],), 1.0, device=fake.device)
        output = dis(fake, fakeHalf).view(-1)
        genLoss = criterion(output, label) + eikonal * eikonalLoss

    # graph: genLoss -> discriminator -> generator
    gradScaler.scale(genLoss).backward()
    gradScaler.step(genOpt)

    for p in dis.parameters():
        p.requires_grad = True

    logData = {}
    logData['generator_loss'] = genLoss.detach()
    logData['eikonal_loss'] = eikonalLoss.detach()
    return logData
