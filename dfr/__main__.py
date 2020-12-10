import torch
import re
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from .hparams import HParams
from .train import train
from .dataset import ImageDataset
from .checkpoint import Checkpoint
from .dataset import makeDataloader

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--data',
        '-d',
        dest='data',
        required=True,
        help='The source directory for the 3D-R2N2 shapenet rendering dataset'
    )
    parser.add_argument(
        '--steps',
        dest='steps',
        default=10 ** 5,
        help='The number of discriminator iterations'
    )
    parser.add_argument(
        '--dlim',
        dest='dlim',
        help='Limit the dataset to the first N items'
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        default=12,
    )
    parser.add_argument(
        '--ckpt',
        '-v',
        dest='ckpt',
        help='Load a checkpoint by name. This will ignore saved hyperparameters in favor of the checkpoint\'s specified values'
    )
    parser.add_argument(
        '--debug-grad',
        dest='debug_grad',
        action='store_true',
        default=False,
        help='Log discriminator gradient data to a Tensorboard histogram. Useful for debugging vanishing/exploding gradients and Lipschitz condition.'
    )
    parser.add_argument(
        '--profile',
        dest='profile',
        action='store_true',
        default=False,
        help='Enable the profiling mode.'
    )
    parser.add_argument(
        '--no-log',
        dest='no_log',
        action='store_true',
        default=False,
        help='Save nothing to disk'
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('Discovered gpu.')
        device = torch.device('cuda')
    else:
        raise Exception('No GPU available! Cannot proceed.')

    runDir = Path.cwd() / 'runs'
    runDir.mkdir(exist_ok=True)
    ckpt = Checkpoint(runDir,
                      version=args.ckpt,
                      epoch=None,
                      device=device,
                      gradientData=args.debug_grad,
                      disableOutput=args.no_log)

    print(ckpt.hparams)
    dataset = ImageDataset(Path(args.data),
                           firstN=int(args.dlim) if args.dlim else None,
                           imageSize=ckpt.gen.frustum.imageSize)

    dataloader = makeDataloader(int(args.batch),
                                dataset,
                                device,
                                workers=0 if args.profile else 1)
    train(dataloader, steps=int(args.steps), ckpt=ckpt)

if __name__ == "__main__":
    main()
