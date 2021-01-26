import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class HParams:
    learningRate: float = 1e-4
    betas: Tuple[int, int] = (0.0, 0.9)
    latentSize: int = 256
    latentStd: float = 2.5
    eikonal: float = 2.0
    r1Factor: float = 10.0
    omega0_first: float = 5.
    omega0_hidden: float = 5.
    sdfWidth: int = 256
    batch: int = 16
    raycast: Tuple[int] = (16, 4)
    # number of input channels for block with input size
    channels = {
        128: 128,
        64: 256,
        32: 384,
        16: 384,
        8: 384,
        4: 384,
        2: 384,
    }

    @property
    def imageSize(self):
        return np.prod(self.raycast)
