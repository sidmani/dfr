import torch
import torch.autograd.profiler as profiler
import numpy as np
import matplotlib.pyplot as plt
from dfr.checkpoint import HParams
from dfr.raycast import raycast, MultiscaleFrustum
from dfr.sdfNetwork import SDFNetwork

# signed-distance function for the half-unit sphere
class MockSDFSphere:
    def __call__(self, x):
        return torch.norm(x, dim=1) - 0.75

class MockSDFCube:
    def __call__(self, x, latents, geomOnly=False):
        global count
        count += x.shape[0]
        box = latents[:, 3:6]
        q = torch.abs(x) - box
        sdf = (torch.norm(torch.clamp(q, min=0.0), dim=1)
                + torch.clamp(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), max=0.0)).unsqueeze(1)
        if geomOnly:
            return sdf
        tx = torch.clamp((x / 2.0) + torch.tensor([0.5, 0.5, 0.5]).view(1, 3), 0.0, 1.0)
        # tx = latents[:, :3]
        return sdf, tx

if __name__ == "__main__":
    count = 0
    phis = torch.tensor([0.0, np.pi/4])
    thetas = torch.tensor([0.0, np.pi/4])
    hp = HParams()
    latents = torch.zeros(2, hp.latentSize)
    latents[0, :6] = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.5, 0.5])
    latents[1, :6] = torch.tensor([1.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    sdf = MockSDFCube()
    frustum = MultiscaleFrustum(hp.fov, [(16, 16), (2, 16), (2, 32)], device=None)

    out, normals = raycast(phis, thetas, frustum, latents, sdf)

    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

    print(f"{count} SDF queries.")
    print(out[0].shape)
    obj1 = out[0].permute(1, 2, 0).detach().numpy()
    obj2 = out[1].permute(1, 2, 0).detach().numpy()
    sil1 = obj1[:, :, 3]
    sil2 = obj2[:, :, 3]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(obj1)
    axs[0, 1].imshow(obj2)
    axs[1, 0].imshow(sil1)
    axs[1, 1].imshow(sil2)

    plt.show()
