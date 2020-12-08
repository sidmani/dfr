from collections import namedtuple

HParams = namedtuple('HParams', [
        'learningRate',
        'betas',
        'weightNorm',
        'discIter',
        'latentSize',
        'latentStd',
        'fov',
        'eikonalFactor',
        'positional',
        'raycastSteps',
    ], defaults=[
        1e-4, # learningRate
        (0.5, 0.9), # betas
        False, # weightNorm
        3, # discIter
        256, # latentSize
        # higher stddev decreases the convergence rate
        # also likely need larger batch size for larger stddev
        0.3, # latent stddev
        0.5, # ~30 deg FOV
        # the eikonal factor has a strong influence on whether initial optimization is
        # done with texture or geometry. Generally we want geometry to be optimized first.
        # If it's too large (around 2.5), the SDF is hard to edit, so textures are modified instead
        # Too low (0.1) and the SDF coalesces slowly or not at all
        1.0, # eikonalFactor
        6, # positional encoding (# of frequencies)
        [(32, 16), (4, 16)], # raycast steps
    ])
