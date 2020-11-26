from dfr.dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt

dataset = ImageDataset(Path('tests/dataset_test/'))
img = dataset[0]

f, axarr = plt.subplots(3, 1)
axarr[0].imshow(img[:3].permute(1, 2, 0).detach().numpy())
axarr[1].imshow(img[3].detach().numpy())
axarr[2].imshow(img[4].detach().numpy())
plt.show()
