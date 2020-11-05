import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=1, fmapSize=64):
        # DC-GAN discriminator architecture
        self.main = nn.Sequential(
            nn.Conv2D(channels, fmapSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fmapSize) x 32 x 32
            nn.Conv2d(fmapSize, fmapSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmapSize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fmapSize*2) x 16 x 16
            nn.Conv2d(fmapSize * 2, fmapSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmapSize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fmapSize*4) x 8 x 8
            nn.Conv2d(fmapSize * 4, fmapSize * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmapSize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fmapSize*8) x 4 x 4
            nn.Conv2d(fmapSize * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
