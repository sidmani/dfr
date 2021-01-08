import torch
import numpy as np
from argparse import ArgumentParser
from dfr.ckpt import Checkpoint
from dfr.__main__ import setArgs
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from dfr.raycast import sample

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)

    count = 12
    hp = ckpt.hparams
    stage = hp.stages[ckpt.startStage]
    sampled = sample(count, device, ckpt, stage.raycast, stage.sharpness)['image']
    downsampled = torch.nn.functional.interpolate(sampled, size=(16, 16), mode='bilinear')
    ckpt.dis.setAlpha(min(1.0, float(ckpt.startEpoch - stage.start) / float(stage.fade)))

    fig, axs = plt.subplots(4, count)
    scores = ckpt.dis(sampled)

    items = []
    for i in range(count):
        img_32 = sampled[i].permute(1, 2, 0).detach().cpu().numpy()
        img_16 = downsampled[i].permute(1, 2, 0).detach().cpu().numpy()
        items.append({'32': img_32, '16': img_16, 'score': scores[i].item()})

    items.sort(key=lambda x: x['score'])
    for idx, item in enumerate(items):
        axs[0, idx].imshow(item['32'])
        axs[1, idx].imshow(item['32'][:, :, 3])
        axs[2, idx].imshow(item['16'])
        axs[3, idx].imshow(item['16'][:, :, 3])
        axs[0, idx].title.set_text(item['score'])

    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
            '--version',
            '-v',
            dest='ckpt',
    )
    parser.add_argument(
            '--epoch',
            '-e',
            type=int
    )

    args = parser.parse_args()
    main(args)
