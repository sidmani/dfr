import torch
from argparse import ArgumentParser
from pathlib import Path
from .train import train, HParams
from .dataset import ImageDataset

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
    args = parser.parse_args()

    print(f"Starting training (${args.steps} steps).")

    if torch.cuda.is_available():
        print('Discovered GPU.')
        device = torch.device('cuda')
    else:
        print('No GPU, falling back to CPU.')
        device = torch.device('cpu')

    hp = HParams(batchSize=int(args.batch))
    print(hp)
    dataCount = int(args.dlim) if args.dlim else None
    dataset = ImageDataset(Path(args.data), hp.imageSize, dataCount)
    train(hp, device, dataset, int(args.steps))
