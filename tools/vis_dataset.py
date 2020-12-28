import torch
from dfr.dataset import ImageDataset, makeDataloader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

dataset = ImageDataset(Path('../cars_128'), sizes=[128, 64, 32])
dataset.requestSizes([128, 64, 32])
# img = dataset[444]
dataloader = makeDataloader(1, dataset, device=torch.device('cpu'))
batch = next(dataloader)

fig, axs = plt.subplots(3)
axs[0].imshow(batch[0][0].permute(1, 2, 0).detach().numpy())
axs[1].imshow(batch[1][0].permute(1, 2, 0).detach().numpy())
axs[2].imshow(batch[2][0].permute(1, 2, 0).detach().numpy())

plt.show()
