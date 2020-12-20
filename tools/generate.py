# query the generator n times, and save the output images

import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from dfr.dataset import ImageDataset
from dfr.ckpt import Checkpoint
from dfr.raycast import raycast

def raycastBatch(batch, device, ckpt):
    phis = torch.ones(batch, device=device) * (np.pi / 6.0)
    thetas = torch.rand_like(phis) * (2.0 * np.pi)
    z = torch.normal(0.0,
                     ckpt.hparams.latentStd,
                     size=(batch, ckpt.hparams.latentSize),
                     device=device)
    fake = raycast(phis, thetas, ckpt.frustum, z, ckpt.gen, ckpt.gradScaler)['image']
    return torch.chunk(fake, fake.shape[0], dim=0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--ckpt',
        '-v',
        dest='ckpt',
        help='Load a checkpoint by name. This will ignore saved hyperparameters in favor of the checkpoint\'s specified values',
        required=True,
    )
    parser.add_argument(
        '-n',
        dest='count',
        help='Number of images to generate',
        default=7500,
    )
    parser.add_argument(
        '-b',
        dest='batch',
        help='Batch size',
        default=12,
    )
    parser.add_argument(
        '-o',
        dest='output',
        help='output directory',
        default='out'
    )
    parser.add_argument(
        '--run-dir',
        '-r',
        dest='runDir',
    )

    args = parser.parse_args()
    device = torch.device('cuda')

    runDir = Path(args.runDir) if args.runDir else Path.cwd() / 'runs'
    ckpt = Checkpoint(runDir, version=args.ckpt, device=device)
    batch = int(args.batch)
    count = int(args.count)
    out = Path(args.output)
    out.mkdir(exist_ok=True)

    print(f'Generating {count} images in {count // batch} batches...')
    for i in tqdm(range(count // batch)):
        c = i * batch
        images = raycastBatch(batch, device, ckpt)
        for img in images:
            save_image(img, out / f'{c:04d}.png')
            c += 1

    if count % batch != 0:
        remaining = count % batch
        images = raycastBatch(remaining, device, ckpt)
        for img in images:
            save_image(img, out / f'{c:04d}.png')
            c += 1

    print(f'Done! Images saved to {out}')
