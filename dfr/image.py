import numpy as np
from kornia.filters import gaussian_blur2d
import torch.nn.functional as F

def blur(img, sigma, kernel=None):
    if kernel is None:
        # rule of thumb for gaussian kernel size is 6*sigma
        kernel = int(np.floor(sigma * 6))
        # kernel should be odd (even kernels modify image size)
        if kernel % 2 == 0:
            kernel += 1

    return gaussian_blur2d(img, (kernel, kernel), (sigma, sigma))

def resample(img, size):
    # note that align_corners=True aligns the centers of the corner pixels
    # so we want align_corners=False, since the discriminator uses average pooling
    return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
