import torch
import numpy as np
from argparse import ArgumentParser
import scipy
from .grad_graph import register_hooks
import scipy.ndimage
from dfr.ckpt import Checkpoint
from dfr.__main__ import setArgs
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from dfr.raycast import sample_like
from dfr.image import blur, resample

def to_np(img):
    return img.permute(1, 2, 0).detach().cpu().numpy()

def main(args):
    device = torch.device('cuda')
    ckpt = Checkpoint(Path.cwd() / 'runs',
                      version=args.ckpt,
                      epoch=args.epoch,
                      device=device)
    count = 12
    hp = ckpt.hparams
    stage = hp.stages[ckpt.startStage]
    size = stage.imageSize
    dataset = ImageDataset(Path('../dataset'))
    dataloader = makeDataloader(count, dataset, device)

    stageIdx = ckpt.startStage
    stages = hp.stages
    dis = ckpt.dis
    dis.setAlpha(stage.evalAlpha(ckpt.startEpoch))

    if stageIdx > 0:
        prevStage = stages[stageIdx - 1]
        sigma = dis.alpha * stage.sigma + (1 - dis.alpha) * prevStage.sigma
    else:
        sigma = stage.sigma

    with torch.no_grad():
        original = next(dataloader)
        blurred = blur(original, sigma)
        real_full = resample(blurred, stage.imageSize)
        real_full.requires_grad = True

        if dis.alpha < 1.:
            real_half = resample(blurred, size=prevStage.imageSize)
            real_half.requires_grad = True
        else:
            real_half = None

    start_fmap = None
    end_fmap = None

    def block_fwd(block, x):
        nonlocal start_fmap
        nonlocal end_fmap
        x = block.layers[0](x)
        x = block.activation(x)
        start_fmap = x
        x = block.layers[2](x)
        x = block.activation(x)
        end_fmap = x
        x = block.layers[4](x)
        return x

    # sample the generator for fake images
    def forward(self, img, half=None):
        # the block corresponding to the current stage
        x = self.adapter[self.stage](img)
        x = self.activation(x)
        x = block_fwd(self.blocks[self.stageCount - self.stage - 1], x)
        # x = self.blocks[self.stageCount - self.stage - 1](x)

        # the faded value from the previous stage
        if self.alpha < 1.0:
            half = torch.nn.functional.avg_pool2d(img, 2)
            x2 = self.adapter[self.stage - 1](half)
            x2 = self.activation(x2)
            # linear interpolation between new & old
            x = (1.0 - self.alpha) * x2 + self.alpha * x

        for block in self.blocks[self.stageCount - self.stage:]:
            x = block(x)

        # no sigmoid, since that's done by BCEWithLogitsLoss
        return self.output(x)

    forward(dis, real_full, None)

    grad_start = torch.autograd.grad(outputs=start_fmap,
            inputs=real_full,
            grad_outputs=torch.ones_like(start_fmap),
            create_graph=True)[0]

    grad_end = torch.autograd.grad(outputs=end_fmap,
            inputs=real_full,
            grad_outputs=torch.ones_like(end_fmap),
            create_graph=True)[0]

    # tmp = full_fmap.mean()
    # get_dot = register_hooks(tmp)
    # tmp.backward()
    # dot = get_dot()
    # dot.save('full_fmap_wo_relu.dot')

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(np.mean(to_np(start_fmap[0]), axis=2))
    axs[1, 0].imshow(np.mean(to_np(end_fmap[0]), axis=2))
    axs[0, 1].imshow(grad_start[0][0].detach().cpu())
    axs[1, 1].imshow(grad_end[0][0].detach().cpu())

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
