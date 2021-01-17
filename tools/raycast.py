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
    out = raycast((phis, thetas), args.resolution, latents, sdf, gradScaler, args.sigma)['full']
    print(out.max())
    print(out.min())

    print(f"{count} SDF queries.")
    obj1 = out[0].permute(1, 2, 0).cpu().detach().numpy()
    obj2 = out[1].permute(1, 2, 0).cpu().detach().numpy()
    sil1 = obj1[:, :, 3]
    sil2 = obj2[:, :, 3]

    fig, axs = plt.subplots(4, 2)
    axs[0, 0].imshow(obj1[:, :, :])
    axs[0, 1].imshow(obj2[:, :, :])
    axs[1, 0].imshow(sil1)
    axs[1, 1].imshow(sil2)

    axs[2, 0].plot(obj1[imgSize // 2, : , 3])
    axs[2, 1].plot(obj2[imgSize // 2, : , 3])

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
            '--sigma',
            type=float,
            default=0.05,
    )
    args = parser.parse_args()
    main(args)
