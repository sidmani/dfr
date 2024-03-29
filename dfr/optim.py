import torch
from torch.cuda.amp import autocast
from .flags import Flags

criterion = torch.nn.BCEWithLogitsLoss()

# R1 gradient penalty (Mescheder et al., 2018)
def R1(real, disReal, gradScaler):
  scaledGrad = torch.autograd.grad(outputs=gradScaler.scale(disReal),
                  inputs=real,
                  grad_outputs=torch.ones_like(disReal),
                  create_graph=True)
  scale = gradScaler.get_scale()
  grad = [g / scale for g in scaledGrad]
  with autocast(enabled=Flags.AMP):
    # grad has shape NCHW so we sum over channel, height, weight dims
    # and take mean over batch (N) dimension
    return (grad[0] ** 2.).sum(dim=[1, 2, 3]).mean()

def stepDiscriminator(real, fake, dis, disOpt, gradScaler, r1Factor):
  disOpt.zero_grad(set_to_none=True)
  real.requires_grad = True
  with autocast(enabled=Flags.AMP):
    disReal = dis(real).view(-1)
    label = torch.full((real.shape[0],), 1., device=disReal.device)
    disLossReal = criterion(disReal, label)

    # the generator's not gonna be updated, so detach it from the grad graph
    disFake = dis(fake.detach()).view(-1)
    label = torch.full((real.shape[0],), 0., device=disReal.device)
    disLossFake = criterion(disFake, label)

  # apply sigmoid to discriminator output, since BCEWithLogitsLoss does that internally
  penalty = r1Factor * R1(real, torch.sigmoid(disReal), gradScaler)

  gradScaler.scale(disLossReal + disLossFake + penalty).backward()
  gradScaler.step(disOpt)

  logData = {}
  with torch.no_grad():
    real_score = torch.sigmoid(disReal).mean().detach()
    fake_score = torch.sigmoid(disFake).mean().detach()
    logData['discriminator_total'] = fake_score - real_score
    logData['penalty'] = penalty.detach()

  return logData

def stepGenerator(fake, normalLength, dis, genOpt, gradScaler, eikonal):
  # save memory by not storing gradients for discriminator
  for p in dis.parameters():
    p.requires_grad = False

  genOpt.zero_grad(set_to_none=True)
  with autocast(enabled=Flags.AMP):
    # the eikonal loss encourages the sdf to have unit gradient
    eikonalLoss = ((normalLength - 1.0) ** 2.0).mean()

    # the discriminator has been updated so we have to run the forward pass again
    # see https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/12
    label = torch.full((fake.shape[0],), 1.0, device=fake.device)
    output = dis(fake).view(-1)
    genLoss = criterion(output, label) + eikonal * eikonalLoss

  # graph: loss -> discriminator -> generator
  gradScaler.scale(genLoss).backward()
  gradScaler.step(genOpt)

  for p in dis.parameters():
    p.requires_grad = True

  return {'eikonal_loss': eikonalLoss.detach()}
