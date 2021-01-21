import torch
import torch.nn.functional as F
from numpy.random import default_rng
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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

        print(f"Loading entire dataset into CPU memory...")
        toTensor = transforms.ToTensor()
        self.rng = default_rng()

        for obj in tqdm(objects):
            with torch.no_grad():
                img = Image.open(obj)
                # some of the images have partially transparent portions (alpha > 0.5)
                # but we don't support that, so make them solid
                # solid = tens[3] > 0.5
                # tens[3, solid] = 1.0
                self.dataset.append(toTensor(img))
                img.close()

    def sample(self, batchSize, res=None):
        idxs = self.rng.choice(len(self.dataset), size=batchSize, replace=False)
        batch = []
        for i in range(batchSize):
            batch.append(self.dataset[idxs[i]])

        with torch.no_grad():
            batchTensor = torch.stack(batch)
            if res is not None:
                # resize the images with bilinear interpolation
                batchTensor = F.interpolate(batchTensor, size=(res, res), mode='bilinear', align_corners=False)

        return batchTensor
