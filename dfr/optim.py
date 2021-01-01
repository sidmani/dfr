import torch
from torch.cuda.amp import autocast

# WGAN-gp gradient penalty
# basic idea: the discriminator should have unit gradient along the real-fake line
def gradientPenalty(dis, realFull, realHalf, fakeFull, fakeHalf, gradScaler):
    with autocast():
        # epsilon different for each batch item
        epsilon = torch.rand(realFull.shape[0], 1, 1, 1, device=realFull.device)
        interpFull = epsilon * realFull + (1.0 - epsilon) * fakeFull
        if dis.alpha < 1.0:
            interpHalf = epsilon * realHalf + (1.0 - epsilon) * fakeHalf
            inputs = (interpFull, interpHalf)
        else:
            interpHalf = None
            inputs = interpFull

        outputs, latest = dis(interpFull, interpHalf, wantsLatest=True)

    # original grad calculation was wrong; see:
    # https://stackoverflow.com/questions/53413706/large-wgan-gp-train-loss
    # grad has shape [batch, channels, px, px]
    scaledGrad = torch.autograd.grad(outputs=gradScaler.scale(outputs),
                               inputs=inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)
    scale = gradScaler.get_scale()
    grad = [p / scale for p in scaledGrad]

    if dis.alpha < 1.:
        # during the fading step, the new block's gradient will explode
        # because it's weighted very low, so its gradient can get large without
        # affecting the overall penalty. So we have to deal with it individually.
        scaledLatestGrad = torch.autograd.grad(outputs=gradScaler.scale(latest),
                                               inputs=interpFull,
                                               grad_outputs=torch.ones_like(latest),
                                               create_graph=True,
                                               retain_graph=True,
                                               only_inputs=True)[0]
        latestGrad = scaledLatestGrad / scale

    with autocast():
        # square; sum over pixel & channel dims; sqrt
        # shape [batch]; each element is the norm of a whole image
        gradNormFull = (grad[0] ** 2.0).sum(dim=[1, 2, 3])
        gradNorm = gradNormFull
        if len(grad) > 1:
            gradNormHalf = (grad[1] ** 2.0).sum(dim=[1, 2, 3])
            gradNorm = gradNorm + gradNormHalf

        if dis.alpha < 1.0:
            gradNormLatest = (latestGrad ** 2.0).sum(dim=[1, 2, 3]).sqrt()
            latestPenalty = ((gradNormLatest - 1.0) ** 2.0).mean()
        else:
            latestPenalty = 0.

        gradNorm = gradNorm.sqrt()
        return ((gradNorm - 1.0) ** 2.0).mean() + latestPenalty
