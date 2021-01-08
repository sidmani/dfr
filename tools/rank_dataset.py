import torch
import numpy as np
from argparse import ArgumentParser
from dfr.ckpt import Checkpoint
from dfr.__main__ import setArgs
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)

    dataset = ImageDataset(Path('../dataset'))
    count = 12
    fig, axs = plt.subplots(4, count)
    items = []
    for i in range(count):
        idx = np.random.randint(0, len(dataset))
        item = dataset[idx].to(device)

        img_16 = torch.nn.functional.interpolate(item.unsqueeze(0), size=(16, 16), mode='bilinear').squeeze(0)
        img_32 = torch.nn.functional.interpolate(item.unsqueeze(0), size=(32, 32), mode='bilinear')
        score = ckpt.dis(img_32).item()

        img_16 = img_16.permute(1, 2, 0).detach().cpu().numpy()
        img_32 = img_32.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        items.append({'32': img_32, '16': img_16, 'score': score, 'idx': idx})


    items.sort(key=lambda x: x['score'])
    for idx, item in enumerate(items):
        axs[0, idx].imshow(item['32'])
        axs[1, idx].imshow(item['32'][:, :, 3])
        axs[2, idx].imshow(item['16'])
        axs[3, idx].imshow(item['16'][:, :, 3])

        axs[1, idx].title.set_text(item['score'])

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
