import torch
import numpy as np
import matplotlib.pyplot as plt
from dfr.checkpoint import HParams
from dfr.generator import Generator
from dfr.raycast.frustum import Frustum

# signed-distance function for the half-unit sphere
class MockSDF:
    def __call__(self, x, latents):
        return torch.norm(x, dim=1) - 0.75

class MockSDFCube:
    def __call__(self, x, latents):
        box = torch.tensor([0.5, 0.5, 0.5])
        q = torch.abs(x) - box
        return (torch.norm(torch.clamp(q, min=0.0), dim=1)
                + torch.clamp(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), max=0.0))

phis = torch.tensor([np.pi/4])
thetas = torch.tensor([np.pi/4])
latents = torch.zeros(1, 256)
hp = HParams(imageSize=128)

frustum = Frustum(2.0 * np.pi / 3.0, hp.imageSize, device=None)
gen = Generator(MockSDFCube(), frustum, hp)
out = gen.raycast(latents, phis, thetas)

obj1 = 1.0 - out[0].detach().numpy()

plt.imshow(obj1, cmap='binary')
plt.show()
