from dfr.dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt

dataset = ImageDataset(Path('tests/dataset_test/'))
img = dataset[0]

plt.imshow(img.permute(1, 2, 0).detach().numpy())
plt.show()
