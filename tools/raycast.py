import torch
from pathlib import Path
from argparse import ArgumentParser
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from dfr.hparams import HParams
from dfr.ckpt import Checkpoint
from dfr.raycast import raycast

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

def main(args):
    device = torch.device('cuda')
    dtype = torch.float
    if args.ckpt:
        ckpt = Checkpoint(Path.cwd() / 'runs',
                          version=args.ckpt,
                          epoch=args.epoch,
                          device=device)
        hp = ckpt.hparams
        sdf = ckpt.gen
        latents = torch.normal(mean=0.0, std=hp.latentStd, size=(2, hp.latentSize), device=device, dtype=dtype)
    else:
        hp = HParams()
        sdf = MockSDFCube()
        latentSize = 6
        latents = torch.zeros(2, latentSize, device=device, dtype=dtype)
        latents[0, :6] = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.5, 0.5], device=device, dtype=dtype)
        latents[1, :6] = torch.tensor([1.0, 0.0, 0.0, 0.5, 0.5, 0.5], device=device, dtype=dtype)

    phis = torch.tensor([0.0, np.pi/4], device=device, dtype=dtype)
    thetas = torch.tensor([0.0, np.pi/4], device=device, dtype=dtype)

    imgSize = np.prod(args.resolution)
    print(f'Raycasting at resolution {imgSize}x{imgSize}')
    gradScaler = GradScaler(enabled=False)
    ret = raycast((phis, thetas), args.resolution, latents, sdf, gradScaler, args.sharpness, halfSharpness=60.)
    out = ret['image']

    print(f"{count} SDF queries.")
    obj1 = out[0].permute(1, 2, 0).cpu().detach().numpy()
    obj2 = out[1].permute(1, 2, 0).cpu().detach().numpy()
    sil1 = obj1[:, :, 3]
    sil2 = obj2[:, :, 3]

    half = ret['half']
    obj1_half = half[0].permute(1, 2, 0).cpu().detach().numpy()
    obj2_half = half[1].permute(1, 2, 0).cpu().detach().numpy()
    sil1_half = obj1_half[:, :, 3]
    sil2_half = obj2_half[:, :, 3]

    fig, axs = plt.subplots(4, 2)
    axs[0, 0].imshow(obj1)
    axs[0, 1].imshow(obj2)
    axs[1, 0].imshow(sil1)
    axs[1, 1].imshow(sil2)
    axs[2, 0].imshow(obj1_half)
    axs[2, 1].imshow(obj2_half)
    axs[3, 0].imshow(sil1_half)
    axs[3, 1].imshow(sil2_half)

    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
            '--version',
            '-v',
            dest='ckpt',
    )
    parser.add_argument(
            '--epoch',
            '-e',
            type=int
    )
    parser.add_argument(
            '--resolution',
            '-r',
            type=int,
            nargs='+',
            default=[32, 4],
    )
    parser.add_argument(
            '--pool',
            type=int,
            default=0,
    )
    parser.add_argument(
            '--sharpness',
            type=float,
            default=10.0,
    )
    args = parser.parse_args()
    main(args)
