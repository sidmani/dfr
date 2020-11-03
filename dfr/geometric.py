import numpy as np
import torch.nn as nn

def geometric_init(layers, dropout=0.0):
    # geometric initialization (SAL section 4)
    # initialize first 7 layers according to thm 1, section 4, SAL
    # note that the skip connection dimension is handled properly
    ps = [1.0 - dropout] * (len(layers) - 2) + [1.0]
    for layer, p in zip(layers[:-1], ps):
        outDim = float(layer.out_features)
        nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(outDim * p))
        nn.init.constant_(layer.bias, 0.0)

    # initialize 8th layer according to part 2 of same thm
    outDim = float(layers[-1].in_features)
    # SAL uses 1e-6, IGR uses 1e-5 for SD
    # SAL also multiplies mean by 2, not sure why
    # the paper says to use constant weights, but SAL code uses gaussian
    nn.init.normal_(layers[-1].weight, np.sqrt(np.pi) / np.sqrt(outDim), 1e-5)
    # R < 0.5 because we're in the unit cube, with some padding to avoid clipping
    nn.init.constant_(layers[-1].bias, -0.45)
