import torch

# WGAN-gp gradient penalty
# basic idea: the discriminator should have unit gradient along the real-fake line
def gradientPenalty(dis, real, fake):
    device = real.device

    epsilon = torch.rand(real.shape[0], 1, 1, device=device)
    interp = epsilon * real + (1.0 - epsilon) * fake
    outputs = dis(interp)
    grad = torch.autograd.grad(outputs=outputs,
                               inputs=interp,
                               grad_outputs=torch.ones(outputs.size(), device=device),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]

    return ((grad.norm(dim=1) - 1.0) ** 2.0).mean()

def stepGenerator(fake, dis, genOpt):
    # check what the discriminator thinks
    genLoss = -dis(fake).mean()

    # graph: genLoss -> discriminator -> generator
    genLoss.backward()
    genOpt.step()

    # reset the discriminator gradients for the discriminator step
    dis.zero_grad(set_to_none=True)

    # reset the generator, since it's done being differentiated
    genOpt.zero_grad(set_to_none=True)

    # the gen's params have changed, so can't backwards again on existing genLoss
    # so we have to run the discriminator again
    # see https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/12
    # discriminator takes only 1/200 the time of the generator pass, so not a problem

def stepDiscriminator(fake, real, dis, disOpt, penaltyWeight=10.0):
    # the generator's not gonna be updated, so detach it from the grad graph
    # detach() sets requires_grad=False, so reset it to True
    fake = fake.detach().requires_grad_()
    # compute the WGAN-gp gradient penalty
    penalty = gradientPenalty(dis, real, fake)

    disLoss = dis(fake).mean() - dis(real).mean() + penalty * penaltyWeight

    disLoss.backward()
    disOpt.step()
    disOpt.zero_grad(set_to_none=True)
