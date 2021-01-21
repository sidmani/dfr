import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Stage:
    start: int
    raycast: list
    batch: int
    fade: int
    discChannels: int
    sigma: float

    @property
    def imageSize(self):
        return np.prod(self.raycast)

    def evalAlpha(self, epoch):
        if self.fade > 0:
            return min(1.0, float(epoch - self.start) / float(self.fade))
        return 1.

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
    stages: Tuple[Stage, ...] = (
        Stage(start=0, raycast=[8], batch=32, fade=0, discChannels=384, sigma=0.1),
        Stage(start=4000, raycast=[16], batch=32, fade=2500, discChannels=384, sigma=0.03),
        Stage(start=7000, raycast=[32], batch=16, fade=2500, discChannels=384, sigma=0.015),
        Stage(start=15000, raycast=[32, 2], batch=16, fade=2500, discChannels=256, sigma=0.0075),
        # Stage(start=6000, raycast=[16], batch=32, fade=5000, discChannels=384, sigma=0.03),
        # Stage(start=20000, raycast=[32], batch=16, fade=5000, discChannels=384, sigma=0.015),
        # Stage(start=35000, raycast=[32, 2], batch=16, fade=5000, discChannels=256, sigma=0.0075),
    )

    def __post_init__(self):
        # sanity checks for the stage parameters
        last = -1
        size = 0
        for idx, stage in enumerate(self.stages):
            # stages must start at increasing times
            assert stage.start > last
            last = stage.start

            if idx == 0:
                assert stage.start == 0
                assert stage.fade == 0
                size = np.prod(stage.raycast)
            else:
                prevStage = self.stages[idx - 1]
                newSize = np.prod(stage.raycast)
                assert newSize == size * 2
                size = newSize

            # fading must end before the next stage
            if len(self.stages) > idx + 1:
                assert stage.fade < self.stages[idx + 1].start
