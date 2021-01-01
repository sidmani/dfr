import torch
from pathlib import Path
from .stats import tensor_stats
from argparse import ArgumentParser
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dfr.ckpt import Checkpoint
from dfr.raycast import sample
from dfr.dataset import ImageDataset, makeDataloader

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)

    # weights = ckpt.dis.currentBlock().layers[0].weight.data
    # weights = ckpt.dis.adapter[ckpt.dis.stage].weight.data
    # print(f'mean 0: {torch.mean(weights)}')
    # print(f'var 0: {torch.var(weights)}')
    dataset = ImageDataset(args.data, [32, 16])
    dataloader = makeDataloader(24, dataset, device)
    batch = next(dataloader)
    batch[0].requires_grad = True
    batch[1].requires_grad = True
    # values = []
    # alphas = []
    ckpt.dis.setStage(1)
    ckpt.dis.setAlpha(0.99)
    out, latest = ckpt.dis(*batch, wantsLatest=True)
    grad = torch.autograd.grad(outputs=latest,
                               inputs=batch[0],
                               grad_outputs=torch.ones_like(latest),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]
    tensor_stats(grad, 'grad')
    tensor_stats(ckpt.dis.latestX, 'values')

    # tensor_stats(weight0.grad, 'grad_weight0')
    # tensor_stats(weight1.grad, 'grad_weight1')
    # block = ckpt.dis.currentBlock()
    # tensor_stats(block.layers[0].bias.data, 'weights')

    # xs = []
    # x2s = []
    # assert ckpt.dis.stage > 0
    # total = 100
    # for i in tqdm(range(51)):
    #     alpha = float(i) / float(total)
    #     alphas.append(alpha)
    #     ckpt.dis.setAlpha(alpha)
    #     value = ckpt.dis(*batch).mean()
    #     # xs.append(hook['x'])
    #     # x2s.append(hook['x2'])
    #     values.append(value.item())
    # plt.plot(alphas, values)

#     x_variances = []
#     x_means = []
#     for x in xs:
#         x_variances.append(torch.var(x))
#         x_means.append(torch.mean(x))

#     x2_variances = []
#     x2_means = []
#     for x2 in x2s:
#         x2_variances.append(torch.var(x))
#         x2_means.append(torch.mean(x))

#     print(f'x_mean: {np.mean(x_means)}')

    # fake = sample(18, device, ckpt, [16, 2])['image']
    # fake16 = torch.nn.functional.avg_pool2d(fake, 2)
    # values_fake = []
    # alphas_fake = []
    # for i in tqdm(range(51)):
    #     alpha = float(i) / float(total)
    #     alphas_fake.append(alpha)
    #     ckpt.dis.setAlpha(alpha)
    #     value = ckpt.dis(fake, fake16).mean().item()
    #     values_fake.append(value)
    # plt.plot(alphas_fake, values_fake, color='red')

    # plt.show()

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
        '--data',
        '-d',
        dest='data',
        required=True,
        type=Path,
        help='The folder of source images',
    )
    args = parser.parse_args()
    main(args)
