from itertools import repeat
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, dataPath):
        super().__init__()
        self.dataset = []

        # load images into RAM
        objects = list(dataPath.glob('*'))
        self.length = len(objects)

        print(f"Loading entire dataset into CPU memory...")
        toTensor = transforms.ToTensor()
        for obj in tqdm(objects):
            img = Image.open(obj)
            self.dataset.append(toTensor(img))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[idx]

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader, device):
    for loader in repeat(dataloader):
        for data in loader:
            yield data.to(device)

def makeDataloader(batchSize, dataset, device, workers=1):
    return iterData(DataLoader(dataset,
            batch_size=batchSize,
            pin_memory=True,
            shuffle=True,
            num_workers=workers), device=device)
