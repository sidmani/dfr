import torch
from dfr.raycast.geometry import rayGrid
import matplotlib.pyplot as plt
import numpy as np

axes = torch.eye(3).unsqueeze(0)

def grid(px, fov):
    D = 1.0 / np.tan(fov / 2.0)
    return rayGrid(axes, px, D)[0].reshape(-1, 3)[:, :2].detach().numpy()

rays_16 = grid(16, 0.5)
rays_32 = grid(32, 0.5)

plt.scatter(rays_16[:, 0], rays_16[:, 1])
plt.scatter(rays_32[:, 0], rays_32[:, 1], color='red')
plt.show()
