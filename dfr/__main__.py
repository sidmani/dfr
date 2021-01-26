import torch
from argparse import ArgumentParser
import pprint
from pathlib import Path
from .train import train
from .ckpt import Checkpoint
from .dataset import ImageDataset, makeDataloader
from .logger import Logger
from .flags import Flags

def setArgs(parser):
    parser.add_argument(
        '--data',
        '-d',
        dest='data',
        required=True,
        type=Path,
        help='The folder of source images',
    )
    parser.add_argument(
        '--fork',
        '-f',
        dest='fork',
        help='fork a previous run'
    )
    parser.add_argument(
        '--epoch',
        '-e',
        dest='epoch',
        type=int,
    )
    parser.add_argument(
        '--steps',
        dest='steps',
        default=10 ** 5,
        type=int,
        help='The number of discriminator iterations',
    )
    parser.add_argument(
        '--ckpt',
        '-v',
        dest='ckpt',
        help='Load a checkpoint by name. This will ignore saved hyperparameters in favor of the checkpoint\'s specified values'
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
    parser.add_argument(
        '--run-dir',
        '-r',
        dest='runDir',
        default=Path.cwd() / 'runs',
        type=Path,
    )
    parser.add_argument(
        '--override-hp',
        dest='override_hp',
        action='store_true',
        default=False
    )

def main(args):
    device = torch.device('cuda')
    args.runDir.mkdir(exist_ok=True)
    ckpt = Checkpoint(args.runDir, version=args.ckpt, device=device, epoch=args.epoch, fork=args.fork, override=args.override_hp)

    Flags.silent = args.no_log
    Flags.profile = args.profile

    if args.no_log:
        logger = None
    else:
        logger = Logger(ckpt)

    pp = pprint.PrettyPrinter(indent=2)
    print('Flags')
    pp.pprint(Flags.__dict__)
    print('Hyperparameters')
    pp.pprint(ckpt.hparams.__dict__)

    # automatically selects best convolution algorithm; yields ~1.5x overall speedup
    torch.backends.cudnn.benchmark = True

    dataset = ImageDataset(args.data)
    dataloader = makeDataloader(dataset, ckpt.hparams.batch, device)
    train(dataloader, device, steps=args.steps, ckpt=ckpt, logger=logger)

if __name__ == "__main__":
    parser = ArgumentParser()
    setArgs(parser)
    main(parser.parse_args())
