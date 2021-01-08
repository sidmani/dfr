import torch
import torch.nn as nn
from torch.nn.functional import conv2d
import numpy as np
from scipy.ndimage import gaussian_filter

class GaussianBlur(nn.Module):
    def __init__(self, kernel, sigma, channels):
        super().__init__()
        assert kernel % 2 == 1
        input = np.zeros((kernel, kernel))
        input[kernel // 2, kernel // 2] = 1.
        filter = gaussian_filter(input, sigma)
        convFilter = torch.from_numpy(filter).float()[None, None, :, :].expand(channels, 1, -1, -1)
        self.register_buffer('filter', convFilter)
        self.channels = channels

    def forward(self, x):
        return conv2d(x, self.filter, groups=self.channels, padding=2)
