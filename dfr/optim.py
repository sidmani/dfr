import torch
from torch.cuda.amp import autocast
import numpy as np

def imageNormSq(t):
    return (t ** 2.0).sum(dim=[1, 2, 3])

def penalty(grad):
    return ((grad - 1.0) ** 2.0).mean()

def penaltyUpper(grad):
    return ((grad.abs() - 1.0).clamp(min=0.0)).mean()

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

        # outputs, fadeFull, fadeHalf = dis(interpFull, interpHalf, intermediate=True)
        outputs = dis(interpFull, interpHalf)

    # original grad calculation was wrong; see:
    # https://stackoverflow.com/questions/53413706/large-wgan-gp-train-loss
    scaledGrad = torch.autograd.grad(outputs=gradScaler.scale(outputs),
                               inputs=inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)
    scale = gradScaler.get_scale()
    grad = [p / scale for p in scaledGrad]

#     if dis.alpha < 0.5:
#         # during the fading step, the new block's gradient will explode
#         # because it's weighted very low, so its gradient can get large without
#         # affecting the overall penalty. So we have to deal with it individually.
#         # scaledFullGrad = torch.autograd.grad(outputs=gradScaler.scale(fadeFull),
#         #                                        inputs=interpFull,
#         #                                        grad_outputs=torch.ones_like(fadeFull),
#         #                                        create_graph=True,
#         #                                        retain_graph=True,
#         #                                        only_inputs=True)[0]
#         # fullGrad = scaledFullGrad / scale
#         # trailingPenalty = 0.
#         # with autocast():
#         #     leadingPenalty = penalty(imageNormSq(fullGrad).sqrt())
#     # elif dis.alpha > 0.5 and dis.alpha < 1.0:
#         # in the latter portion of the fading stage, the adapter's gradient will explode
#         # scaledHalfGrad = torch.autograd.grad(outputs=gradScaler.scale(fadeHalf),
#         #                                        inputs=interpHalf,
#         #                                        grad_outputs=torch.ones_like(fadeHalf),
#         #                                        create_graph=True,
#         #                                        retain_graph=True,
#         #                                        only_inputs=True)[0]
#         # halfGrad = scaledHalfGrad / scale
#         # leadingPenalty = 0.
#         # with autocast():
#         # leadingPenalty = (0.82 - (0.82 - 0.735) * dis.alpha) * imageNormSq(fadeFull).sqrt().mean() * 0.01
#         leadingPenalty = 0.0
#         # leadingPenalty = (1.0 - dis.alpha) * 0.01 * penaltyUpper(imageNormSq(fullGrad).sqrt())
#         # trailingPenalty = 1e-7 * penalty(imageNormSq(halfGrad).sqrt())
#         trailingPenalty = 0.
#     else:
#         leadingPenalty = 0.
#         trailingPenalty = 0.

    with autocast():
        gradNorm = imageNormSq(grad[0])
        if len(grad) > 1:
            # gradNormHalf = (grad[1] ** 2.0).sum(dim=[1, 2, 3])
            gradNorm = gradNorm + imageNormSq(grad[1])

        return penalty(gradNorm.sqrt())
    # if dis.alpha < 1.0:
    #     gradNormLatest = (latestGrad ** 2.0).sum(dim=[1, 2, 3]).sqrt()
    #     latestPenalty = ((gradNormLatest - 1.0) ** 2.0).mean()
    # else:
    #     latestPenalty = 0.
    # return ((gradNorm - 1.0) ** 2.0).mean() + latestPenalty
    # return penalty(gradNorm.sqrt()) + leadingPenalty + trailingPenalty
