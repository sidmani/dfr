import torch
from dfr.dataset import ImageDataset
from pathlib import Path

def test_dataset():
    d = ImageDataset(Path('tests/dataset_test'), imageSize=8)
    assert len(d) == 2

    t = d[0]
    assert t.shape == (3, 8, 8)
