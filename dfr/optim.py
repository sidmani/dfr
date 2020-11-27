import torch

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

def stepDiscriminator(fake, real, dis, disOpt):
    # the generator's not gonna be updated, so detach it from the grad graph
    # also possible that generator has been modified in-place, so can't backprop through it
    # detach() sets requires_grad=False, so reset it to True
    # need to clone so that in-place ops in CNN are legal
    fake = fake.detach().clone().requires_grad_()

    disFake = dis(fake).mean()
    disReal = dis(real).mean()
    disLoss = disFake - disReal

    disLoss.backward()
    disOpt.step()
    disOpt.zero_grad(set_to_none=True)
    return {'discriminator_real': disReal,
            'discriminator_fake': disFake,
            'discriminator_total': disLoss}
