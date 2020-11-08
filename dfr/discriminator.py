import torch
import torch.nn as nn

# DCGAN weight initialization
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     else:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, channels=1, fmapSize=64):
        super().__init__()
        # DC-GAN discriminator architecture
        # batch norm omitted per WGAN-GP
        self.main = nn.Sequential(
            nn.Conv2d(channels, fmapSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fmapSize, fmapSize * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fmapSize * 2, fmapSize * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fmapSize * 4, fmapSize * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fmapSize * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # weight initialization
        # self.main.apply(weights_init)

    def forward(self, x):
        # re-insert the channel dimension
        x = x.unsqueeze(1)
        return self.main(x).squeeze()
