import torch
import numpy as np
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

dataset = ImageDataset(Path('../cars_128'))
idx = np.random.randint(0, len(dataset))
print(idx)
item = dataset[idx]
img_128 = torch.nn.functional.interpolate(item.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
img_64 = torch.nn.functional.interpolate(item.unsqueeze(0), size=(64, 64), mode='bilinear').squeeze(0)
img_32 = torch.nn.functional.interpolate(item.unsqueeze(0), size=(32, 32), mode='bilinear').squeeze(0)
img_128 = img_128.permute(1, 2, 0).detach().numpy()
img_64 = img_64.permute(1, 2, 0).detach().numpy()
img_32 = img_32.permute(1, 2, 0).detach().numpy()


fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(img_128)
axs[1, 0].imshow(img_64)
axs[2, 0].imshow(img_32)

axs[0, 1].imshow(img_128[:, :, 3])
axs[1, 1].imshow(img_64[:, :, 3])
axs[2, 1].imshow(img_32[:, :, 3])

plt.show()
