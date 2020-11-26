import torch
import numpy as np
import matplotlib.pyplot as plt
from dfr.checkpoint import HParams
from dfr.raycast.frustum import Frustum
from dfr.raycast import raycast
from dfr.sdfNetwork import SDFNetwork

count = 0
class MockSDFCube:
    def __call__(self, x, latents, geomOnly=False):
        global count
        count += x.shape[0]
        box = torch.tensor([0.5, 0.5, 0.5])
        q = torch.abs(x) - box
        sdf = (torch.norm(torch.clamp(q, min=0.0), dim=1)
                + torch.clamp(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), max=0.0)).unsqueeze(1)
        if geomOnly:
            return sdf
        tx = (x / 2.0) + torch.tensor([0.5, 0.5, 0.5]).view(1, 3)
        tx = torch.clamp(tx, min=0.0)
        return sdf, tx

phis = torch.tensor([0.0, np.pi/4])
thetas = torch.tensor([0.0, np.pi/4])
hp = HParams(imageSize=64, weightNorm=False)
latents = torch.normal(mean=0.0, std=0.1, size=(2, hp.latentSize))
frustum = Frustum(hp.fov, hp.imageSize, device=None)
sdf = MockSDFCube()

out, normals = raycast(phis, thetas, latents, frustum, sdf, hp.raySamples)
print(f"{count} SDF queries")

obj1 = out[0].permute(1, 2, 0).detach().numpy()
obj2 = out[1].permute(1, 2, 0).detach().numpy()

fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(obj1[:, :, :3])
axs[0, 1].imshow(obj1[:, :, 3])
axs[0, 2].imshow(obj1[:, :, 4])
axs[1, 0].imshow(obj2[:, :, :3])
axs[1, 1].imshow(obj2[:, :, 3])
axs[1, 2].imshow(obj2[:, :, 4])
plt.show()
