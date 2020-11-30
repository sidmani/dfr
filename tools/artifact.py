import torch
import numpy as np
import matplotlib.pyplot as plt
from dfr.checkpoint import HParams
from dfr.raycast import raycast, MultiscaleFrustum
from dfr.sdfNetwork import SDFNetwork

class CapturePts:
    def __init__(self):
        self.pts = []

    def __call__(self, x, latents, geomOnly=False):
        if geomOnly:
            self.pts.append(x)
        box = torch.tensor([0.5, 0.5, 0.5])
        q = torch.abs(x) - box
        sdf = (torch.norm(torch.clamp(q, min=0.0), dim=1)
                + torch.clamp(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), max=0.0)).unsqueeze(1)
        if geomOnly:
            return sdf
        tx = (x / 2.0) + torch.tensor([0.5, 0.5, 0.5]).view(1, 3)
        return sdf, tx

batch = 2
hp = HParams()
latents = torch.normal(mean=0.0, std=0.1, size=(batch, hp.latentSize))
frustum = MultiscaleFrustum(hp.fov, [(16, 16), (2, 16), (2, 32)], device=None)
sdf = CapturePts()

for i in range(2):
    phis = torch.zeros(batch)
    thetas = torch.tensor([0.0, 0.0])

    out, normals = raycast(phis, thetas, frustum, latents, sdf)

allPts = torch.cat(sdf.pts).detach().numpy()

axis = 1
smallZ = np.logical_and(allPts[:, axis] <= 0.53, allPts[:, axis] >= 0.49)
fig = plt.figure(figsize=(64, 64))
ax = fig.add_subplot(111)
ax.axis('equal')
ax.scatter(allPts[smallZ, (axis + 1) % 3], allPts[smallZ, (axis + 2) % 3])
plt.show()
