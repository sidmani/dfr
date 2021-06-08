import torch
from .raycast import raycast
from tqdm import tqdm
from .optim import stepDiscriminator, stepGenerator
from .flags import Flags
from .precondition import precondition

def train(dataloader, device, steps, ckpt, logger):
  # if this is a new version, precondition the SDF
  if ckpt.startEpoch == 0:
    print('Preconditioning SDF...')
    precondition(ckpt, device, logger=logger, steps=1500)

  print('Starting training loop...')
  for epoch in tqdm(range(ckpt.startEpoch, steps), initial=ckpt.startEpoch, total=steps):
    loop(dataloader, device, ckpt, logger, epoch)

# separate the loop function to make sure all variables go out of scope
# otherwise memory may not be freed, causing 2x max memory usage
def loop(dataloader, device, ckpt, logger, epoch):
  dis, gen, disOpt, genOpt, gradScaler = ckpt.dis, ckpt.gen, ckpt.disOpt, ckpt.genOpt, ckpt.gradScaler
  hparams = ckpt.hparams

  with torch.no_grad():
    real = next(dataloader).to(device)
    s = hparams.imageSize
    real = torch.nn.functional.interpolate(real, size=(s, s), mode='bilinear', align_corners=False)
    real.requires_grad = True

  # sample the generator for fake images
  batch = real.shape[0]
  angles = hparams.sampleAngles(batch, device)
  z = hparams.sampleLatents(batch, device)
  fake, normalLength = raycast(angles, z, hparams.raycast, gen, gradScaler)
  logData = {'fake': fake, 'real': real}

  disData = stepDiscriminator(real, fake, dis, disOpt, gradScaler, hparams.r1Factor)
  logData.update(disData)

  genData = stepGenerator(fake, normalLength, dis, genOpt, gradScaler, hparams.eikonal)
  logData.update(genData)

  gradScaler.update()

  if logger is not None:
    logger.log(logData, epoch)

  if not Flags.silent and epoch % 100 == 0:
    ckpt.save(epoch, overwrite=True)
