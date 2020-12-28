import torch
from dfr.dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

dataset = ImageDataset(Path('../cars_128'), firstN=5)
img = dataset[0]

# bilinear = transforms.Resize((32, 32))(img)
bilinear = transforms.functional.resize(img, [32, 32])
bilinear_twice = transforms.functional.resize(img, [64, 64])
bilinear_twice = transforms.functional.resize(bilinear_twice, [32, 32])
# avgPool = torch.nn.functional.avg_pool2d(img, 4)

fig, axs = plt.subplots(3)
axs[0].imshow(img.permute(1, 2, 0).detach().numpy())
axs[0].title.set_text('original')

axs[1].imshow(bilinear.permute(1, 2, 0).detach().numpy())
axs[1].title.set_text('bilinear')

axs[2].imshow(bilinear_twice.permute(1, 2, 0).detach().numpy())
axs[2].title.set_text('bilinear_twice')

plt.show()
