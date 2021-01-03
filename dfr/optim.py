import torch
from torch.cuda.amp import autocast
import numpy as np

def imageNorm(t):
    # sqrt has NaN subgradient at 0, so reshape + norm instead
    # https://github.com/pytorch/pytorch/issues/6394
    flattened = t.view(t.shape[0], -1)
    return torch.norm(flattened, dim=1)

# WGAN-gp gradient penalty
# basic idea: the discriminator should have unit gradient along the real-fake line
def gradientPenalty(dis, realFull, realHalf, fakeFull, fakeHalf, gradScaler):
    with autocast():
        # epsilon different for each batch item
        epsilon = torch.rand(realFull.shape[0], 1, 1, 1, device=realFull.device)
        interpFull = epsilon * realFull + (1.0 - epsilon) * fakeFull
        inputs = interpFull

        outputs = dis(interpFull, None)

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

    with autocast():
        gradNormSq = (grad[0] ** 2.0).sum(dim=[1, 2, 3])
        return ((gradNormSq.sqrt() - 1.0) ** 2.0).mean()
