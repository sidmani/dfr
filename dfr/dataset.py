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
            # TODO: bilinear interpolation infrequently causes artifacts in the output
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
        for obj in tqdm(objects):
            img = Image.open(obj)
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

def makeDataloader(batchSize, dataset, device, workers=1):
    # num_workers=1 because dataset is already in RAM
    return iterData(DataLoader(dataset,
            batch_size=batchSize,
            pin_memory=True,
            shuffle=True,
            num_workers=workers), device=device)
