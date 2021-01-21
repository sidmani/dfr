import torch
import numpy as np
from dfr.dataset import ImageDataset, solidify
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional_tensor import gaussian_blur
from dfr.image import blur

dataset = ImageDataset(Path('../dataset'))

img_128 = dataset.sample(1, 128)[0]
img_64 = dataset.sample(1, 64)[0]
img_32 = dataset.sample(1, 32)[0]
img_16 = dataset.sample(1, 16)[0]
img_128 = img_128.permute(1, 2, 0).detach().numpy()
img_64 = img_64.permute(1, 2, 0).detach().numpy()
img_32 = img_32.permute(1, 2, 0).detach().numpy()
img_16 = img_16.permute(1, 2, 0).detach().numpy()

fig, axs = plt.subplots(4, 3)
axs[0, 0].imshow(img_128[:, :, :3])
axs[1, 0].imshow(img_64[:, :, :3])
axs[2, 0].imshow(img_32[:, :, :3])
axs[3, 0].imshow(img_16[:, :, :3])

axs[0, 1].imshow(img_128[:, :, 3])
axs[1, 1].imshow(img_64[:, :, 3])
axs[2, 1].imshow(img_32[:, :, 3])
axs[3, 1].imshow(img_16[:, :, 3])

axs[0, 2].plot(img_128[64, :, 3])
axs[1, 2].plot(img_64[32, :, 3])
axs[2, 2].plot(img_32[16, :, 3])
axs[3, 2].plot(img_16[8, :, 3])

plt.show()
