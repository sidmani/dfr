from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pathlib import Path
from dfr.gan import GAN
from dfr.dataset import DFRDataModule

if __name__ == "__main__":
    imageSize = 64
    parser = ArgumentParser()
    parser.add_argument(
        '--gpus',
        dest='gpu',
        default=0,
    )
    parser.add_argument(
        '--data',
        dest='data',
        default='tests/dataset_test',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        default=4,
    )

    args = parser.parse_args()

    dataset = DFRDataModule(int(args.batch),
                            Path(args.data),
                            imageSize=imageSize,
                            workers=1)
    model = GAN(imageSize=imageSize)
    trainer = Trainer(gpus=args.gpu,
                      automatic_optimization=False,
                      # precision=(32 if args.full_prec else 16),
                      max_epochs=2)
    trainer.fit(model, dataset)
