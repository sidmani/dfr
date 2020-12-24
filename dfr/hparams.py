import numpy as np
from collections import namedtuple

Stage = namedtuple('Stage', ['start', 'raycast', 'batch', 'fade', 'discChannels'])

HParams = namedtuple('HParams', [
        'learningRate',
        'betas',
        'discIter',
        'latentSize',
        'latentStd',
        'fov',
        'eikonalFactor',
        'illumFactor',
        'sineOmega',
        'sdfWidth',
        'trainingStages',
    ], defaults=[
        1e-4, # learningRate
        (0.5, 0.9), # betas
        1, # discIter
        256, # latentSize
        # higher stddev decreases the convergence rate
        # also likely need larger batch size for larger stddev
        3.0, # latent stddev
        0.5, # ~30 deg FOV
        1.0, # eikonalFactor
        0.0, # illum factor
        1.0, # the SIREN omega_0 value
        512,
        [
            Stage(0, [16], 24, fade=0, discChannels=384),
            Stage(20000, [16, 2], 18, fade=10000, discChannels=384),
            Stage(60000, [32, 2], 12, fade=10000, discChannels=256),
            Stage(90000, [32, 4], 6, fade=10000, discChannels=128),
        ]
    ])

def checkStages(stages):
    last = -1
    size = 0
    for idx, stage in enumerate(stages):
        # stages must start at increasing times
        assert stage.start > last
        last = stage.start

        if idx == 0:
            size = np.prod(stage.raycast)
        else:
            newSize = np.prod(stage.raycast)
            assert newSize == size * 2
            size = newSize

        # fading must end before the next stage
        if len(stages) > idx + 1:
            assert stage.fade < stages[idx + 1].start
