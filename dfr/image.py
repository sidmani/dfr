import numpy as np
from torchvision.transforms.functional_tensor import gaussian_blur

def blur(img, sigma, kernel=None):
    if kernel is None:
        # rule of thumb for gaussian kernel size is 6*sigma
        kernel = int(np.floor(sigma * 6))
        # kernel should be odd (even kernels modify image size)
        if kernel % 2 == 0:
            kernel += 1

    return gaussian_blur(img, [kernel, kernel], [sigma, sigma])
