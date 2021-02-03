import torch
from .flags import Flags
from itertools import repeat
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ImageDataset:
  def __init__(self, dataPath):
    super().__init__()
    self.dataset = []

    objects = list(dataPath.glob('*'))
    self.length = len(objects)

    print("Loading entire dataset into CPU memory...")
    toTensor = transforms.ToTensor()

    with torch.no_grad():
      for obj in tqdm(objects):
        img = Image.open(obj)
        self.dataset.append(toTensor(img))
        img.close()

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return self.dataset[idx]

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader):
  for loader in repeat(dataloader):
    for data in loader:
      yield data

def makeDataloader(dataset, batch):
  return iterData(DataLoader(dataset,
      batch_size=batch,
      pin_memory=True,
      shuffle=True,
      num_workers=0 if Flags.profile else 1))
