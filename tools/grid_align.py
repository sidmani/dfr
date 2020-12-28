import torch
from dfr.raycast.ray import makeRays
import matplotlib.pyplot as plt
import numpy

axes = torch.eye(3).unsqueeze(0)
fov = 0.5
D = 5.0
rays_16 = makeRays(axes, 16, D, fov, torch.float)[0].reshape(-1, 3)[:, :2].detach().numpy()
rays_32 = makeRays(axes, 32, D, fov, torch.float)[0].reshape(-1, 3)[:, :2].detach().numpy()
rays_64 = makeRays(axes, 64, D, fov, torch.float)[0].reshape(-1, 3)[:, :2].detach().numpy()

plt.scatter(rays_16[:, 0], rays_16[:, 1])
plt.scatter(rays_32[:, 0], rays_32[:, 1], color='red')
plt.scatter(rays_64[:, 0], rays_64[:, 1], color='green')
plt.show()
