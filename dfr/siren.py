import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega.

    # If is_first=True, omega is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega=30.0):
        super().__init__()
        self.omega = omega
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                siren_linear_init(self.linear, self.omega)

    def forward(self, input, gamma, beta):
        return torch.sin(self.omega * (self.linear(input) * gamma + beta))

def siren_linear_init(layer, hidden_omega):
    with torch.no_grad():
        layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / hidden_omega,
                               np.sqrt(6 / layer.in_features) / hidden_omega)
