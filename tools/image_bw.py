import torch
import numpy as np
from argparse import ArgumentParser
import scipy
import scipy.ndimage
from dfr.ckpt import Checkpoint
from dfr.__main__ import setArgs
from dfr.dataset import ImageDataset
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from dfr.raycast import sample_like
from dfr.image import blur, resample

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)
    count = 24
    hp = ckpt.hparams
    stage = hp.stages[ckpt.startStage]
    size = stage.imageSize
    dataset = ImageDataset(Path('../dataset'))

    stageIdx = ckpt.startStage
    stages = hp.stages
    dis = ckpt.dis
    # dis.setAlpha(stage.evalAlpha(ckpt.startEpoch))
    dis.setAlpha(1.)

    sigma = stage.sigma

    with torch.no_grad():
        real_full = dataset.sample(count, stage.imageSize).to(device)
        # original = next(dataloader)
        # real_full = resample(original, stage.imageSize)
        real_full = real_full[:, 3, :, :].unsqueeze(1)
        real_full.requires_grad = True

    # sample the generator for fake images
    sampled = sample_like(real_full, ckpt, stage.raycast, sigma)
    fake = sampled['full'][:, 3, :, :].unsqueeze(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    disReal = dis(real_full).view(-1)
    label = torch.full((real_full.shape[0],), 1.0, device=disReal.device)
    disLossReal = criterion(disReal, label)

    label = torch.full((real_full.shape[0],), 0.0, device=disReal.device)
    disFake = dis(fake).view(-1)
    disLossFake = criterion(disFake, label)

    loss = disLossReal + disLossFake

    if args.dataset:
        inputs = real_full
    else:
        inputs = fake

    grad = torch.autograd.grad(outputs=loss,
                                inputs=inputs,
                                grad_outputs=torch.ones_like(loss),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)

    items = []
    for i in range(count):
        if args.dataset:
            target = real_full
            scores = disReal
        else:
            target = fake
            scores = disFake

        img_32 = target[i].permute(1, 2, 0).detach().cpu().numpy()

        grad_32 = torch.clamp(grad[0][i], min=-1., max=1.).permute(1, 2, 0).detach().cpu().numpy()
        fft = scipy.fft.fft2(img_32, axes=[0, 1])
        fft_shifted = scipy.fft.fftshift(fft)
        fft_mag = np.log10(np.linalg.norm(fft_shifted, axis=2))
        items.append({'32': img_32, 'grad': grad_32, 'fft':fft_mag, 'score': scores[i].item()})

    fig, axs = plt.subplots(6, 6)

    items.sort(key=lambda x: x['score'])
    axs[0, 0].set_ylabel('image_rgb', size='large')
    axs[1, 0].set_ylabel(f'image_ch{args.channel}', size='large')
    axs[2, 0].set_ylabel(f'illum', size='large')
    axs[3, 0].set_ylabel('disc_grad_rgb', size='large')
    axs[4, 0].set_ylabel(f'disc_grad_ch{args.channel}', size='large')
    axs[5, 0].set_ylabel('fft', size='large')
    for idx, item in enumerate(items[:6]):
        # axs[0, idx].imshow(item['32'][:, :, :3])
        axs[1, idx].imshow(item['32'])
        # if not args.dataset:
        #     axs[2, idx].imshow(item['illum'])
        # axs[3, idx].imshow((1 + 1000 * item['grad'][:, :, :3]) / 2.)
        axs[4, idx].imshow(item['grad'])
        axs[5, idx].imshow(item['fft'])
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
