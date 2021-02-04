from .flags import Flags
from itertools import repeat
import multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset:
  def __init__(self, dataPath):
    super().__init__()
    self.objects = list(dataPath.glob('*'))
    self.toTensor = transforms.ToTensor()

  def __len__(self):
    return len(self.objects)

  def __getitem__(self, idx):
    with Image.open(self.objects[idx]) as img:
      return self.toTensor(img)

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader):
  for loader in repeat(dataloader):
    for data in loader:
      yield data

def makeDataloader(dataset, batch, workers=min(multiprocessing.cpu_count(), 8)):
  return iterData(DataLoader(dataset,
      batch_size=batch,
      pin_memory=True,
      shuffle=True,
      num_workers=0 if Flags.profile else workers))
