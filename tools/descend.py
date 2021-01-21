import torch
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import matplotlib.animation as animation
import scipy
import scipy.ndimage
from dfr.ckpt import Checkpoint
from dfr.__main__ import setArgs
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from dfr.raycast import sample
from dfr.image import blur, resample

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)
    count = 4
    hp = ckpt.hparams
    stage = hp.stages[ckpt.startStage]
    size = stage.imageSize
    stageIdx = ckpt.startStage
    stages = hp.stages
    dis = ckpt.dis
    dis.setAlpha(stage.evalAlpha(ckpt.startEpoch))

    sigma = stage.sigma

    # sample the generator for fake images
    sampled = sample(count, device, ckpt, stage.raycast, sigma)
    fake = sampled['full'].detach()
    fake.requires_grad = True
    criterion = torch.nn.BCEWithLogitsLoss()
    before = fake.permute(0, 2, 3, 1).detach().cpu().numpy()
    images = []
    score_0 = None

    for i in tqdm(range(1000)):
        dis.zero_grad()
        score = dis(fake).squeeze()
        if score_0 is None:
            score_0 = score
        label = torch.full((count,), 1.0, device=device)
        loss = criterion(score, label)
        loss.backward()
        with torch.no_grad():
            fake = fake - 0.05 * fake.grad
            fake.grad = None
        fake.requires_grad = True
        image = fake.permute(0, 2, 3, 1).detach().cpu().numpy()

    fig, axs = plt.subplots(8, count)
    for i in range(count):
        axs[0, i].title.set_text(score_0[i].item())
        axs[0, i].imshow(before[i])
        axs[1, i].imshow(before[i][:, :, 0], vmin=-0.1, vmax=1.1)
        axs[2, i].imshow(before[i][:, :, 3], vmin=-0.1, vmax=1.1)
        axs[3, i].title.set_text(score[i].item())
        axs[3, i].imshow(image[i])
        axs[4, i].imshow(image[i][:, :, 0], vmin=-0.1, vmax=1.1)
        axs[5, i].imshow(image[i][:, :, 3], vmin=-0.1, vmax=1.1)
        axs[6, i].imshow(image[i][:, :, 0] - before[i][:, :, 0], vmin=-0.1, vmax=1.1)
        axs[7, i].imshow(image[i][:, :, 3] - before[i][:, :, 3], vmin=-0.1, vmax=1.1)

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
    parser.add_argument(
            '--channel',
            '-c',
            type=int,
            default=3,
    )
    parser.add_argument(
            '--dataset',
            action='store_true',
            default=False
    )
    args = parser.parse_args()
    main(args)
