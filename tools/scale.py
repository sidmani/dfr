import torch
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms.functional import resize
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from dfr.hparams import HParams
from dfr.ckpt import Checkpoint
from dfr.raycast import raycast
from dfr.image import blur

count = 0

# signed-distance function for the half-unit sphere
class MockSDFSphere:
    def __call__(self, x):
        return torch.norm(x, dim=1) - 0.75

class MockSDFCube:
    def __call__(self, x, latents, mask, geomOnly=False):
        latents = latents[mask]
        global count
        count += x.shape[0]
        box = latents[:, 3:6]
        q = torch.abs(x) - box
        sdf = (torch.norm(torch.clamp(q, min=0.0), dim=1)
                + torch.clamp(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), max=0.0)).unsqueeze(1)
        if geomOnly:
            return sdf
        tx = torch.clamp((x / 2.0) + torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype).view(1, 3), 0.0, 1.0)
        return sdf, tx

def main():
    device = torch.device('cuda')
    dtype = torch.float

    hp = HParams()
    sdf = MockSDFCube()
    latentSize = 6
    latents = torch.zeros(1, latentSize, device=device, dtype=dtype)
    latents[0, :6] = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.5, 0.5], device=device, dtype=dtype)

    phis = torch.tensor([0], device=device, dtype=dtype)
    thetas = torch.tensor([0], device=device, dtype=dtype)
    gradScaler = GradScaler(enabled=False)
    real = raycast((phis, thetas), [32, 4], latents, sdf, gradScaler, 0.)['full']

    stages = hp.stages
    resolutions = [s.raycast for s in stages]
    sigmas = [s.sigma for s in stages]

    fig, axs = plt.subplots(6, len(resolutions))
    for i, (res, sigma) in enumerate(zip(resolutions, sigmas)):
        ret = raycast((phis, thetas), res, latents, sdf, gradScaler, sigma)['full']
        ret = ret[0].permute(1, 2, 0).cpu().detach().numpy()
        axs[0, i].imshow(ret)
        axs[1, i].imshow(ret[:, :, 3])

        size = np.prod(res)
        out = resize(real, size=(size, size), interpolation=Image.BILINEAR)[0]
        # out = torch.nn.functional.interpolate(real, size=(size, size), mode='nearest')[0]
        out = out.permute(1, 2, 0).cpu().detach().numpy()
        axs[3, i].imshow(out)
        axs[4, i].imshow(out[:, :, 3])

        axs[2, i].imshow(ret[:, :, 3] - out[:, :, 3])

    plt.show()

if __name__ == "__main__":
    main()
