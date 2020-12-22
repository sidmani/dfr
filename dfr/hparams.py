from collections import namedtuple

Stage = namedtuple('Stage', ['start', 'raycast', 'batch'])

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
        3, # discIter
        256, # latentSize
        # higher stddev decreases the convergence rate
        # also likely need larger batch size for larger stddev
        3.0, # latent stddev
        0.5, # ~30 deg FOV
        # the eikonal factor has a strong influence on whether initial optimization is
        # done with texture or geometry. Generally we want geometry to be optimized first.
        # If it's too large (around 2.5), the SDF is hard to edit, so textures are modified instead
        # Too low (0.1) and the SDF coalesces slowly or not at all
        50.0, # eikonalFactor
        0.0, # illum factor
        1.0, # the SIREN omega_0 value
        512,
        [
            Stage(0, [32, 2], 12),
            Stage(300, [32, 2], 12),
            Stage(600, [32, 2], 12),
            # Stage(5000, [32, 2], 18),
            # Stage(40000, [32, 4], 9),
        ]
    ])
