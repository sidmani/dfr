import torch
import re
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from .train import train
from .dataset import ImageDataset
from .checkpoint import Checkpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--data',
        '-d',
        dest='data',
        required=True,
    )
    parser.add_argument(
        '--steps',
        dest='steps',
        default=10 ** 5,
    )
    parser.add_argument(
        '--dlim',
        dest='dlim',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        default=6,
    )
    parser.add_argument(
        '--ckpt',
        '-v',
        dest='ckpt',
    )
    parser.add_argument(
        '--debug-grad',
        dest='debug_grad',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('Discovered GPU.')
        device = torch.device('cuda')
    else:
        print('No GPU, falling back to CPU.')
        device = torch.device('cpu')

    runDir = Path.cwd() / 'runs'
    runDir.mkdir(exist_ok=True)
    ckpt = Checkpoint(runDir,
                      version=args.ckpt,
                      epoch=None,
                      device=device,
                      gradientData=args.debug_grad)

    # don't have the explicit image size, so compute it from the raycast scales
    imageSize = np.prod([item[0] for item in ckpt.hparams.raycastSteps])
    dataset = ImageDataset(Path(args.data),
                           firstN=int(args.dlim) if args.dlim else None,
                           imageSize=imageSize)

    train(batchSize=int(args.batch),
          device=device,
          dataset=dataset,
          steps=int(args.steps),
          ckpt=ckpt)
