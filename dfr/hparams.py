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
    sigma: float # sigma for gaussian filter

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
        # rule of thumb for sharpness is 2.5 * resolution, except first step
        Stage(start=0, raycast=[8], batch=32, fade=0, discChannels=384, sigma=1.0),
        # Stage(start=8000, raycast=[8, 2], batch=32, fade=7500, discChannels=384, sigma=1.5),
        Stage(start=6000, raycast=[8, 2], batch=32, fade=2500, discChannels=384, sigma=2),
        Stage(start=9000, raycast=[8, 2, 2], batch=16, fade=5000, discChannels=384, sigma=1),
        # Stage(start=25000, raycast=[8, 2, 2], batch=16, fade=7500, discChannels=384, sharpness=24., sigma=0.75),
        # Stage(start=35000, raycast=[16, 2, 2], batch=16, fade=8000, discChannels=256, sharpness=160.),
        # Stage(start=60000, raycast=[32, 4], batch=8, fade=8000, discChannels=128, sharpness=320.),
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
                assert stage.fade == 0
                size = np.prod(stage.raycast)
            else:
                # sharpness fading must start after previous stage fading ends
                prevStage = self.stages[idx - 1]
                # assert stage.start - (prevStage.start + prevStage.fade) > self.sharpnessFadeIn
                newSize = np.prod(stage.raycast)
                assert newSize == size * 2
                size = newSize

            # fading must end before the next stage
            if len(self.stages) > idx + 1:
                assert stage.fade < self.stages[idx + 1].start
