import torch.nn as nn
import numpy as np

# geometric initialization (SAL section 4)
def geometricInit(layers, r=0.5):
    # initialize first 7 layers according to thm 1, section 4, SAL
    # note that the skip connection dimension is handled properly
    for layer in layers[:-1]:
        outDim = float(layer.out_features)
        nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(outDim))
        nn.init.constant_(layer.bias, 0.0)

    # initialize 8th layer according to part 2 of same thm
    outDim = float(layers[-1].in_features)

    # SAL uses 1e-6, IGR uses 1e-5 for SD
    # SAL also multiplies mean by 2, not sure why
    # the paper says to use constant weights, but SAL code uses gaussian
    nn.init.normal_(layers[-1].weight, np.sqrt(np.pi) / np.sqrt(outDim), 1e-5)
    nn.init.constant_(layers[-1].bias, -r)
