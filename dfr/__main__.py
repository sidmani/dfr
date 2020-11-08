from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from .dataset import DataModule, ImageDataset

def main(args):
    logger = TensorBoardLogger(name='lightning_logs', save_dir=Path.cwd())
    dataset = ImageDataset(Path(args.data), imageSize=64)
    trainer = Trainer(gpus=args.gpu,
                      logger=logger,
                      automatic_optimization=False,
                      precision=(32 if args.full_prec else 16),
                      max_epochs=int(args.max_epochs))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--32',
        dest='full_prec',
        action='store_true',
        default=False,
        help='use 32-bit training (default 16-bit AMP)'
    )
    parser.add_argument(
        '--data',
        '-d',
        dest='data',
        required=True,
    )
    parser.add_argument(
        '--gpus',
        dest='gpu',
    )
    parser.add_argument(
        '--epochs',
        dest='max_epochs',
        default=3000,
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        default=48,
    )

    args = parser.parse_args()
    main(args)
