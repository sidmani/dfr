import torch
from torch.cuda.amp import autocast

# R1 gradient penalty (Mescheder et al., 2018)
def R1(real, disReal, gradScaler):
    scaledGrad = torch.autograd.grad(outputs=gradScaler.scale(disReal),
                                     inputs=real,
                                     grad_outputs=torch.ones_like(disReal),
                                     create_graph=True,
                                     retain_graph=True,
                                     only_inputs=True)[0]
    scale = gradScaler.get_scale()
    grad = scaledGrad / scale
    with autocast():
        # note that grad has shape NCHW
        # so we sum over channel, height, weight dims
        # and take mean over batch (N) dimension
        return (grad ** 2.0).sum(dim=[1, 2, 3]).mean()
