import torch
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from .util import rand_range

@dataclass
class HParams:
  learningRate: float = 1e-4
  betas: Tuple[int, int] = (0.0, 0.9)
  latentSize: int = 256
  latentStd: float = 2.5
  eikonal: float = 2.0
  r1Factor: float = 10.0
  omega_first: float = 5.
  omega_hidden: float = 15.
  sdfWidth: int = 256
  batch: int = 16
  raycast: Tuple[int] = (16, 4)

  # angle ranges
  azimuth: Tuple[float] = (0, 0)
  elevation: Tuple[float] = (0, 0)
  # azimuth: Tuple[float] = (0, np.pi * 2)
  # elevation: Tuple[float] = (10 * np.pi / 180, 30 * np.pi / 180)

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

  def sampleLatents(self, count, device):
    return torch.normal(0.0, self.latentStd, (count, self.latentSize), device=device)

  def sampleAngles(self, count, device):
    phis = rand_range(self.elevation, (count,), device)
    thetas = rand_range(self.azimuth, (count,), device)
    return (phis, thetas)

  @property
  def imageSize(self):
    return np.prod(self.raycast)
