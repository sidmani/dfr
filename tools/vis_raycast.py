import torch
import numpy as np
import matplotlib.pyplot as plt
from dfr.checkpoint import HParams
from dfr.raycast.frustum import Frustum
from dfr.raycast import raycast
from dfr.sdfNetwork import SDFNetwork
from dfr.texture import TextureNetwork

# signed-distance function for the half-unit sphere
class MockSDFSphere:
    def __call__(self, x):
        return torch.norm(x, dim=1) - 0.75

class MockSDFCube:
    def __call__(self, x):
        box = torch.tensor([0.5, 0.5, 0.5])
        q = torch.abs(x[:, :3]) - box
        return (torch.norm(torch.clamp(q, min=0.0), dim=1)
                + torch.clamp(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), max=0.0)).unsqueeze(1)

class MockTexture:
    def __call__(self, x):
        return (x[:, :3] / 2.0) + torch.tensor([0.5, 0.5, 0.5]).view(1, 3)

phis = torch.tensor([0.0, 0.0])
thetas = torch.tensor([0.0, 0.0])
hp = HParams(imageSize=64, weightNorm=False)
latents = torch.normal(mean=0.0, std=1e-2, size=(2, hp.latentSize))
frustum = Frustum(hp.fov, hp.imageSize, device=None)
# sdf = MockSDFCube()
# texture = MockTexture()
sdf = SDFNetwork(hp)
texture = TextureNetwork(hp)

out, normals = raycast(phis, thetas, latents, frustum, sdf, texture, hp.raySamples)

obj1 = out[0].permute(1, 2, 0).detach().numpy()
obj2 = out[1].permute(1, 2, 0).detach().numpy()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(obj1)
axs[1].imshow(obj2)
plt.show()

