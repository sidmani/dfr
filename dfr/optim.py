import torch

# WGAN-gp gradient penalty
# basic idea: the discriminator should have unit gradient along the real-fake line
def gradientPenalty(dis, real, fake):
    device = real.device

    # epsilon different for each batch item
    # ignoring that torch.rand is in [0, 1), but wgan-gp specifies [0, 1]
    epsilon = torch.rand(real.shape[0], 1, 1, 1, device=device)
    interp = epsilon * real + (1.0 - epsilon) * fake
    outputs = dis(interp)

    # original grad calculation was wrong; see:
    # https://stackoverflow.com/questions/53413706/large-wgan-gp-train-loss
    # grad has shape [batch, channels, px, px]
    grad = torch.autograd.grad(outputs=outputs,
                               inputs=interp,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]

    # square; sum over pixel & channel dims; sqrt
    # shape [batch]; each element is the norm of a whole image
    gradNorm = (grad ** 2.0).sum(dim=[1, 2, 3]).sqrt()
    return ((gradNorm - 1.0) ** 2.0).mean(), { 'gradient_norm': gradNorm.detach() }

def stepGenerator(fake, normals, dis, genOpt, eikonalFactor):
    # the eikonal loss encourages the sdf to have unit gradient
    eikonalLoss = ((normals.norm(dim=1) - 1.0) ** 2.0).mean()

    # check what the discriminator thinks
    genLoss = -dis(fake).mean() + eikonalFactor * eikonalLoss

    for p in dis.parameters():
        p.requires_grad = False

    # graph: genLoss -> discriminator -> generator
    genLoss.backward()
    genOpt.step()

    for p in dis.parameters():
        p.requires_grad = True

    # reset the discriminator gradients for the discriminator step
    # dis.zero_grad(set_to_none=True)

    # reset the generator, since it's done being differentiated
    genOpt.zero_grad(set_to_none=True)

    # the gen's params have changed, so can't backwards again on existing genLoss
    # so we have to run the discriminator again
    # see https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/12
    # discriminator takes only 1/200 the time of the generator pass, so not a problem
    return {'generator_loss': genLoss,
            'eikonal_loss': eikonalLoss}

def stepDiscriminator(fake, real, dis, disOpt, penaltyWeight=10.0):
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    fake = fake.detach().clone().requires_grad_()

    # compute the WGAN-gp gradient penalty
    penalty, logData = gradientPenalty(dis, real, fake)

    disFake = dis(fake).mean()
    disReal = dis(real).mean()
    disLoss = disFake - disReal + penalty * penaltyWeight

    disLoss.backward()
    disOpt.step()
    disOpt.zero_grad(set_to_none=True)
    return {'discriminator_real': disReal,
            'discriminator_fake': disFake,
            'discriminator_total': disLoss,
            **logData}
