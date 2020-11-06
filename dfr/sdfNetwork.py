import torch
import torch.nn as nn
import numpy as np

# geometric initialization (SAL section 4)
def geometric_init(layers, r=0.98, dropout=0.0):
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
    nn.init.constant_(layers[-1].bias, -r)

class SDFNetwork(nn.Module):
    def __init__(self, latentSize=256, width=512, weightNorm=False):
        super().__init__()
        assert width > latentSize + 3

        self.layers = nn.ModuleList([
            nn.Linear(latentSize + 3, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (latentSize + 3)),
            # skip connection from input: latent + x (into 5th layer)
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        # SAL geometric initialization
        geometric_init(self.layers)

        if weightNorm:
            for i in range(8):
                self.layers[i] = nn.utils.weight_norm(self.layers[i])

        self.activation = nn.ReLU

    def forward(self, x):
        r = x
        for i in range(7):
            if i == 4:
                # skip connection
                # per SAL supplementary: divide by sqrt(2) to normalize
                r = torch.cat([x, r], dim=1) / np.sqrt(2)
            r = self.activation(self.layers[i](r))

        # DeepSDF uses tanh, but SALD says it's optional
        # empirically doesn't make much difference
        return torch.tanh(self.layers[7](r))
