import torch
from argparse import ArgumentParser
import pprint
import numpy as np
from pathlib import Path
from .train import train
from .ckpt import Checkpoint
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
        '--debug-grad',
        dest='debug_grad',
        action='store_true',
        default=False,
        help='Log discriminator gradient data to a Tensorboard histogram. Useful for debugging vanishing/exploding gradients and Lipschitz condition.'
    )
    parser.add_argument(
        '--debug-activations',
        '-A',
        dest='debug_act',
        action='store_true',
        default=False,
        help='Log the activation histogram of each layer.'
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
        logger = Logger(ckpt,
                        gradientData=args.debug_grad,
                        activations=args.debug_act)

    pp = pprint.PrettyPrinter(indent=2)
    print('Flags')
    pp.pprint(Flags.__dict__)
    print('Hyperparameters')
    pp.pprint(ckpt.hparams.__dict__)

    # selects best convolution algorithm; yields ~1.5x overall speedup
    torch.backends.cudnn.benchmark = True
    train(args.data, device, steps=args.steps, ckpt=ckpt, logger=logger)

if __name__ == "__main__":
    parser = ArgumentParser()
    setArgs(parser)
    main(parser.parse_args())
