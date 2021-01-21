import torch
import numpy as np
from dfr.dataset import ImageDataset, solidify
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional_tensor import gaussian_blur
from dfr.image import blur

dataset = ImageDataset(Path('../dataset'))
# idx = np.random.randint(0, len(dataset))
# item = dataset[idx]
# item = dataset.sample(1)

# item = blur(item.unsqueeze(0), 1.).squeeze(0)


# img_64 = torch.nn.functional.interpolate(item.unsqueeze(0), size=(64, 64), mode='nearest')
# img_64_up = torch.nn.functional.interpolate(img_64, size=(128, 128), mode='nearest').squeeze(0)
# diff = img_128 - img_64_up

# fig, axs = plt.subplots(4, 1)
# axs[0].imshow(img_128.permute(1, 2, 0).detach().numpy())
# axs[1].imshow(img_64.squeeze(0).permute(1, 2, 0).detach().numpy())
# axs[2].imshow(img_64_up.permute(1, 2, 0).detach().numpy())
# axs[3].imshow(diff.permute(1, 2, 0).detach().numpy())

# img_128 = torch.nn.functional.interpolate(item, size=(128, 128), mode='bilinear')
# img_128 = solidify(img_128, threshold=0.5).squeeze(0)
# img_64 = torch.nn.functional.interpolate(item, size=(64, 64), mode='bilinear').squeeze(0)
# img_32 = torch.nn.functional.interpolate(item, size=(32, 32), mode='bilinear').squeeze(0)
# img_16 = torch.nn.functional.interpolate(item, size=(16, 16), mode='bilinear').squeeze(0)
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
