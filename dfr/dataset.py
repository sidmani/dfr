import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, dataPath, px, firstN=None):
        super().__init__()

        pipeline = transforms.Compose([
            transforms.Resize((px, px)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        blur = transforms.GaussianBlur(3.0, sigma=0.4)

        self.dataset = []
        objects = list(dataPath.glob('*'))
        if firstN is not None and firstN < len(objects):
            objects = objects[:firstN]

        # load images into RAM
        print(f"Loading dataset ({len(objects)} objects) into RAM...")
        imgsPerFolder = 24
        for folder in tqdm(objects):
            # pick a random view (1 per object)
            idx = np.random.randint(0, imgsPerFolder)
            img = Image.open(folder / 'rendering' / f"{idx:02d}.png")
            # image to tensor, and invert (1.0 is white)
            tens = 1.0 - pipeline(img)
            # mask the salient portion
            mask = tens < 1.0
            tens[mask] = 0.0
            # gaussian blur to mimic soft shading
            tens = blur(tens)
            self.dataset.append(tens.squeeze(0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
