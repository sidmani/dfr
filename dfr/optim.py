import torch
from torch.cuda.amp import autocast, GradScaler

# WGAN-gp gradient penalty
# basic idea: the discriminator should have unit gradient along the real-fake line
def gradientPenalty(dis, real, fake, gradScaler):
    # epsilon different for each batch item
    # ignoring that torch.rand is in [0, 1), but wgan-gp specifies [0, 1]
    with autocast():
        epsilon = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
        interp = epsilon * real + (1.0 - epsilon) * fake
        outputs = dis(interp)

    # original grad calculation was wrong; see:
    # https://stackoverflow.com/questions/53413706/large-wgan-gp-train-loss
    # grad has shape [batch, channels, px, px]
    scaledGrad = torch.autograd.grad(outputs=gradScaler.scale(outputs),
                               inputs=interp,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]

    grad = scaledGrad / gradScaler.get_scale()

    with autocast():
        # square; sum over pixel & channel dims; sqrt
        # shape [batch]; each element is the norm of a whole image
        gradNorm = (grad ** 2.0).sum(dim=[1, 2, 3]).sqrt()
        return ((gradNorm - 1.0) ** 2.0).mean()

def stepGenerator(fake, normals, illum, dis, genOpt, eikonalFactor, illumFactor, gradScaler):
    with autocast():
        # normals have already been scaled to correct values
        # the eikonal loss encourages the sdf to have unit gradient
        eikonalLoss = ((normals.norm(dim=1) - 1.0) ** 2.0).mean()

        # normals should not be pointing away from the view direction (i.e. illum < 0.5)
        # since the raycaster is pretty much perfect, this should even out the weird bits
        illumLoss = (0.5 - illum.clamp(max=0.5)).mean()

        # check what the discriminator thinks
        genLoss = -dis(fake).mean() + eikonalFactor * eikonalLoss + illumFactor * illumLoss

    for p in dis.parameters():
        p.requires_grad = False

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
    return {'generator_loss': genLoss.detach(),
            'eikonal_loss': eikonalLoss.detach(),
            'illum_loss': illumLoss.detach()}

def stepDiscriminator(fake, real, dis, disOpt, gradScaler):
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    fake = fake.detach().clone().requires_grad_()

    penalty = gradientPenalty(dis, real, fake, gradScaler)

    with autocast():
        disFake = dis(fake).mean()
        disReal = dis(real).mean()
        disLoss = disFake - disReal + penalty * 10.0

    gradScaler.scale(disLoss).backward()
    gradScaler.step(disOpt)

    disOpt.zero_grad(set_to_none=True)

    return {'discriminator_real': disReal.detach(),
            'discriminator_fake': disFake.detach(),
            'discriminator_total': disLoss.detach()}
