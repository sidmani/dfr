import torch
import numpy as np
from argparse import ArgumentParser
from dfr.ckpt import Checkpoint
from dfr.__main__ import setArgs
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from dfr.raycast import sample
from dfr.image import blur

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)
    count = 6
    hp = ckpt.hparams
    dataset = ImageDataset(Path('../dataset'))
    dataloader = makeDataloader(dataset, count, device)
    dis = ckpt.dis

    with torch.no_grad():
        real = next(dataloader)
        # real = blur(real, 3.)
        s = hp.imageSize
        real = torch.nn.functional.interpolate(real, size=(s, s), mode='bilinear')
        real.requires_grad = True

    # sample the generator for fake images
    sampled = sample(count, device, ckpt, hp.raycast, 0.)
    fake = sampled['full']
    criterion = torch.nn.BCEWithLogitsLoss()

    disReal = dis(real).view(-1)
    label = torch.full((real.shape[0],), 1.0, device=disReal.device)
    disLossReal = criterion(disReal, label)

    label = torch.full((real.shape[0],), 0.0, device=disReal.device)
    disFake = dis(fake).view(-1)
    disLossFake = criterion(disFake, label)

    loss = disLossReal + disLossFake

    if args.dataset:
        inputs = real
    else:
        inputs = fake

    grad = torch.autograd.grad(outputs=loss,
                                inputs=inputs,
                                grad_outputs=torch.ones_like(loss),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)

    items = []
    for i in range(fake.shape[0]):
        if args.dataset:
            target = real
            scores = disReal
        else:
            target = fake
            scores = disFake
        img_32 = target[i].permute(1, 2, 0).detach().cpu().numpy()
        grad_32 = torch.clamp(grad[0][i], min=-1., max=1.).permute(1, 2, 0).detach().cpu().numpy()
        items.append({'32': img_32, 'grad': grad_32, 'score': scores[i].item()})

    fig, axs = plt.subplots(4, count)

    items.sort(key=lambda x: x['score'])
    axs[0, 0].set_ylabel('image_rgb', size='large')
    axs[1, 0].set_ylabel(f'image_ch{args.channel}', size='large')
    axs[2, 0].set_ylabel('disc_grad_rgb', size='large')
    axs[3, 0].set_ylabel(f'disc_grad_ch{args.channel}', size='large')
    for idx, item in enumerate(items):
        axs[0, idx].title.set_text(item['score'])
        axs[0, idx].imshow(item['32'][:, :, :3])
        axs[1, idx].imshow(item['32'][:, :, args.channel])
        axs[2, idx].imshow((1 + 1000 * item['grad'][:, :, :3]) / 2.)
        axs[3, idx].imshow(item['grad'][:, :, args.channel])

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
