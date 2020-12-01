import numpy as np
from itertools import repeat
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, dataPath, firstN=None, imageSize=64):
        super().__init__()
        self.imageSize = imageSize

        pipeline = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
        ])

        self.dataset = []
        objects = sorted(list(dataPath.glob('*')))

        # firstN limits the dataset size if present
        if firstN is not None and firstN < len(objects):
            objects = objects[:firstN]

        if len(objects) == 0:
            raise Exception('Dataset is empty!')

        # load images into RAM
        print(f"Loading dataset ({len(objects)} objects) into RAM...")
        imgsPerFolder = 24
        for folder in tqdm(objects):
            # pick a random view (1 per object)
            idx = np.random.randint(0, imgsPerFolder)
            img = Image.open(folder / 'rendering' / f"{idx:02d}.png")
            self.dataset.append(pipeline(img))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader, device):
    for loader in repeat(dataloader):
        for data in loader:
            yield data.to(device)

def makeDataloader(batchSize, dataset, device):
    # num_workers=1 because dataset is already in RAM
    return iterData(DataLoader(dataset,
            batch_size=batchSize,
            pin_memory=True,
            shuffle=True,
            num_workers=1), device=device)
