import torch
from .flags import Flags
from itertools import repeat
from torch.utils.data.dataset import Dataset
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from numpy.random import default_rng
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from .image import blur

# threshold an image with alpha into solid and transparent portions
# def solidify(image, alphaChannel=3, threshold=0.5):
#     solid = (image[:, alphaChannel] > threshold).float()
#     image[:, alphaChannel] = solid
#     image *= solid.unsqueeze(1)
#     return image

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

#     def sample(self, batchSize, res=None, sigma=0.05):
#         # get a batch of images by index without replacement
#         idxs = self.rng.choice(len(self.dataset), size=batchSize, replace=False)
#         batch = [self.dataset[i] for i in idxs]

#         with torch.no_grad():
#             batchTensor = torch.stack(batch)
#             if sigma > 0:
#                 # gaussian blur as a low-pass filter (sigma is in sdf units so convert to pixels)
#                 batchTensor = blur(batchTensor, sigma * batchTensor.shape[2] / 2)

#             if res is not None:
#                 # resize the images with bilinear interpolation
#                 batchTensor = F.interpolate(batchTensor, size=(res, res), mode='bilinear', align_corners=False)

#         return batchTensor

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader, device):
    for loader in repeat(dataloader):
        for data in loader:
            yield data.to(device)

def makeDataloader(dataset, batch, device):
    return iterData(DataLoader(dataset,
            batch_size=batch,
            pin_memory=True,
            shuffle=True,
            num_workers=0 if Flags.profile else 1), device=device)
