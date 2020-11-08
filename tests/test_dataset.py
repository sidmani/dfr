import torch
from dfr.dataset import ImageDataset
from pathlib import Path

def test_dataset():
    d = ImageDataset(Path('tests/dataset_test'), 8)
    assert len(d) == 2

    t = d[0]
    assert t.shape == (8, 8)
