import torch
from argparse import ArgumentParser
from pathlib import Path
from .train import train
from .dataset import ImageDataset
import re

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
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('Discovered GPU.')
        device = torch.device('cuda')
    else:
        print('No GPU, falling back to CPU.')
        device = torch.device('cpu')

    if args.ckpt:
        # load the newest checkpoint for given version
        checkpointPath = Path.cwd() / 'runs' / f"v{args.ckpt}"
        if not checkpointPath.exists:
            raise Exception(f'Version {args.ckpt} does not exist')

        available = list(checkpointPath.glob('*.pt'))
        if len(available) == 0:
            raise Exception(f'No checkpoints found for version {args.ckpt}')

        nums = []
        for f in available:
            match = re.match("e([0-9]+)", str(f.stem))
            nums.append(int(match[1]))

        checkpoint = torch.load(checkpointPath / f"e{max(nums)}.pt")
        version = int(args.ckpt)
        print(f"Loaded version {version}, epoch {checkpoint['epoch']}.")
    else:
        checkpointDir = Path.cwd() / 'runs'
        checkpointDir.mkdir(parents=True, exist_ok=True)
        versions = [-1]
        for f in checkpointDir.glob('v*'):
            match = re.match('v([0-9]+)', str(f.stem))
            versions.append(int(match[1]))

        version = max(versions) + 1
        checkpoint = None
        print(f"This is version {version}.")

    dataCount = int(args.dlim) if args.dlim else None
    train(int(args.batch),
          device,
          Path(args.data),
          dataCount,
          int(args.steps),
          version,
          checkpoint)
