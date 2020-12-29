from itertools import repeat
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, dataPath, sizes):
        super().__init__()

        self.dataset = {}
        for size in sizes:
            self.dataset[size] = []
        self.requested = sizes

        # load images into RAM
        objects = list(dataPath.glob('*'))
        self.length = len(objects)

        print(f"Loading entire dataset into CPU memory...")
        toTensor = transforms.ToTensor()
        for obj in tqdm(objects):
            img = Image.open(obj)
            for size in sizes:
                # *very* important to use a good resampling filter
                # don't use lanczos; causes ringing artifacts and mixes channels
                # PIL does low-pass prefiltering and torchvision doesn't, so use PIL
                # also see https://stackoverflow.com/q/60949936
                resized = img.resize((size, size), resample=Image.BILINEAR)
                self.dataset[size].append(toTensor(resized))

    def requestSizes(self, sizes):
        self.requested = sizes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [self.dataset[x][idx] for x in self.requested]

# infinite dataloader
# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567
def iterData(dataloader, device):
    for loader in repeat(dataloader):
        for data in loader:
            yield [t.to(device) for t in data]

def makeDataloader(batchSize, dataset, device, workers=1):
    # num_workers=1 because dataset is already in RAM
    return iterData(DataLoader(dataset,
            batch_size=batchSize,
            pin_memory=True,
            shuffle=True,
            num_workers=workers), device=device)
