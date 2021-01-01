import torch
from pathlib import Path
from argparse import ArgumentParser
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from dfr.ckpt import Checkpoint
from dfr.raycast import raycast
from torchvision import transforms

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)

    phis = torch.ones(12, device=device) * (np.pi / 6.0)
    thetas = torch.rand_like(phis) * (2.0 * np.pi)
    z = torch.normal(0.0,
                     ckpt.hparams.latentStd,
                     size=(12, ckpt.hparams.latentSize),
                     device=device)
    batch_16 = raycast(phis, thetas, [16, 2], ckpt.hparams.fov, z, ckpt.gen, ckpt.gradScaler)['image']
    batch_32 = raycast(phis, thetas, [16, 4], ckpt.hparams.fov, z, ckpt.gen, ckpt.gradScaler)['image']
    batch_32_down = torch.nn.functional.avg_pool2d(batch_32, 2)
    # batch_32_down = torch.nn.functional.interpolate(batch_32, scale_factor=0.5)

    dis_16 = ckpt.dis(batch_16, None)
    dis_32 = ckpt.dis(batch_32_down, None)
    print(f'Mean discriminator distance: {torch.abs(dis_16 - dis_32).mean().item()}')

    obj1 = batch_16[0].permute(1, 2, 0).cpu().detach().numpy()
    obj2 = batch_32_down[0].permute(1, 2, 0).cpu().detach().numpy()
    sil1 = obj1[:, :, 3]
    sil2 = obj2[:, :, 3]

    print(f'Mean visual distance: {np.abs(obj1 - obj2).mean()}')

    # fig, axs = plt.subplots(2, 3)
    # axs[0, 0].imshow(obj1)
    # axs[0, 1].imshow(obj2)
    # axs[1, 0].imshow(sil1)
    # axs[1, 1].imshow(sil2)
    # axs[0, 2].imshow(np.linalg.norm(obj1 - obj2, axis=2))
    # axs[1, 2].imshow(np.abs(sil1 - sil2))
    # plt.show()

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
    args = parser.parse_args()
    main(args)
