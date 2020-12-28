import torch
from pathlib import Path
from argparse import ArgumentParser
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from dfr.hparams import HParams
from dfr.ckpt import Checkpoint
from dfr.raycast import raycast
from dfr.sdfNetwork import SDFNetwork
from torchvision import transforms

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
        # tx = latents[:, :3]
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
        sdf = ckpt.gen.sdf
        latents = torch.normal(mean=0.0, std=hp.latentStd, size=(2, hp.latentSize), device=device, dtype=dtype)
    else:
        hp = HParams()
        sdf = MockSDFCube()
        latentSize = 6
        latents = torch.zeros(1, latentSize, device=device, dtype=dtype)
        latents[0, :6] = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.5, 0.5], device=device, dtype=dtype)

    phis = torch.tensor([np.pi/4], device=device, dtype=dtype)
    thetas = torch.tensor([np.pi/4], device=device, dtype=dtype)
    scaler = GradScaler(init_scale=32768.)

    ret_1 = raycast(phis, thetas, [32], hp.fov, latents, sdf, scaler)['image'][0]
    ret_2 = raycast(phis, thetas, [32, 4], hp.fov, latents, sdf, scaler)['image'][0]
    # avgPooled = torch.nn.functional.avg_pool2d(ret_2, 4)
    # bilinear = transforms.functional.resize(ret_2, [32, 32])
    print(ret_2.shape)
    bilinear = torch.nn.functional.interpolate(ret_2.unsqueeze(0), size=[32, 32], mode='bilinear').squeeze(0)
    # print((avgPooled - bilinear).abs().mean().item())

    obj1 = ret_1.permute(1, 2, 0).cpu().detach().numpy()
    obj2 = bilinear.permute(1, 2, 0).cpu().detach().numpy()
    sil1 = obj1[:, :, 3]
    sil2 = obj2[:, :, 3]

    print(f'Mean distance: {np.abs(obj1 - obj2).mean()}')

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(obj1)
    axs[0, 1].imshow(obj2)
    axs[1, 0].imshow(sil1)
    axs[1, 1].imshow(sil2)
    axs[0, 2].imshow(np.linalg.norm(obj1 - obj2, axis=2))
    axs[1, 2].imshow(np.abs(sil1 - sil2))
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
    args = parser.parse_args()
    main(args)
