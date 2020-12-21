import torch
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from .train import train
from .dataset import ImageDataset, makeDataloader
from .ckpt import Checkpoint
from .logger import Logger
from tools.memory import print_memory_stats

def setArgs(parser):
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
    )

def main(args):
    runDir = Path(args.runDir) if args.runDir else Path.cwd() / 'runs'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('No GPU available! Cannot proceed.')

    runDir.mkdir(exist_ok=True)
    ckpt = Checkpoint(runDir, version=args.ckpt, device=device, noLog=args.no_log)

    if args.no_log:
        logger = None
    else:
        logger = Logger(ckpt,
                        gradientData=args.debug_grad,
                        activations=args.debug_act)

    print(ckpt.hparams)
    dataset = ImageDataset(Path(args.data),
                           firstN=int(args.dlim) if args.dlim else None,
                           imageSize=np.prod(ckpt.hparams.raycastSteps))

    dataloader = makeDataloader(int(args.batch),
                                dataset,
                                device,
                                workers=0 if args.profile else 1)
    train(dataloader, steps=int(args.steps), ckpt=ckpt, logger=logger)

if __name__ == "__main__":
    parser = ArgumentParser()
    setArgs(parser)
    args = parser.parse_args()
    main(args)
    print_memory_stats()
