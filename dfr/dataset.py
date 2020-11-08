import torch
import numpy as np
import multiprocessing
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, dataPath, px):
        super().__init__()

        pipeline = transforms.Compose([
            transforms.Resize((px, px)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.dataset = []

        # load images into RAM
        for folder in tqdm(list(dataPath.glob('*'))):
            images = list(folder.glob('rendering/*.png'))
            # pick a random view (1 per object)
            idx = np.random.randint(0, len(images))
            img = Image.open(images[idx])
            # drop the channels dimension
            tens = 1.0 - pipeline(img).squeeze(0)
            # invert and mask the salient portion
            mask = tens < 1.0
            tens[mask] = 0.0
            self.dataset.append(tens)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class DataModule(LightningDataModule):
    def __init__(self,
                 batchSize,
                 dataPath,
                 imageSize,
                 workers=multiprocessing.cpu_count() - 1):
        super().__init__()
        self.workers = workers
        self.batchSize = batchSize
        self.dataset = ImageDataset(dataPath, imageSize)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batchSize,
                          pin_memory=True,
                          shuffle=True,
                          # need different seeds for different workers
                          # otherwise may return identical batches
                          worker_init_fn=lambda i: np.random.seed((torch.initial_seed() + i) % 2 ** 32),
                          num_workers=self.workers)
