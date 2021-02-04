import numpy as np
from kornia.filters import gaussian_blur2d

def blur(img, sigma, kernel=None):
    if kernel is None:
        # rule of thumb for gaussian kernel size is 6*sigma
        kernel = int(np.floor(sigma * 6))
        # kernel should be odd (even kernels modify image size)
        if kernel % 2 == 0:
            kernel += 1

        if kernel == 1:
            return img
            # raise Exception('Kernel is 1! Are you sure?')

    return gaussian_blur2d(img, (kernel, kernel), (sigma, sigma))
