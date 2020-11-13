import torch
import torch.nn as nn
import numpy as np
from .raycast.frustum import enumerateRays, sphereToRect
from .raycast.shader import fastRayIntegral, shade
from .raycast.sample import sampleUniform, sampleStratified, scaleRays

class Generator(nn.Module):
    def __init__(self, sdf, frustum, hparams):
        super().__init__()

        self.sdf = sdf
        self.frustum = frustum
        self.hparams = hparams

    def raycast(self, latents, phis, thetas):
        device = latents.device

        # build a rotated frustum for each input angle
        rays = enumerateRays(phis, thetas, self.frustum.viewField, self.hparams.imageSize)

        # uniformly sample distances from the camera in the unit sphere
        # TODO: should sampling be weighted by ray length?
        # unsqueeze because we're using the same sample values for all objects
        samples = sampleStratified(
                self.frustum.near,
                self.frustum.far,
                self.hparams.raySamples).unsqueeze(0)

        # unscaled camera location
        cameraLoc = sphereToRect(phis, thetas, self.frustum.cameraD)

        # compute the sampling points for each ray that intersects the unit sphere
        targets = scaleRays(
                rays[:, self.frustum.mask],
                samples[:, self.frustum.mask],
                self.frustum.cameraD * cameraLoc)

        # compute intersections for rays
        values, texture, normals = fastRayIntegral(latents, targets, self.sdf, 10e-10)

        # shape [batch, px, px, channels]
        result = torch.zeros(*rays.shape[:3], 3, device=device)
        shaded = shade(values, texture, normals)
        result[:, self.frustum.mask] = shaded.view(result.shape[0], -1, 3)
        return result.permute(0, 3, 1, 2)

    def sample(self, batchSize, phi=np.pi / 6.0, device=None):
        # elevation angle: phi = pi/6
        phis = torch.ones(batchSize, device=device) * phi
        # azimuthal angle: 0 <= theta < 2pi
        thetas = torch.rand(batchSize, device=device) * (2.0 * np.pi)
        # latents with mean 0, sigma 1e-4 (per SALD)
        # DFR uses sigma=sqrt(0.33), but that's a different architecture (OccNet)
        z = torch.normal(
                mean=0.0,
                std=1e-2,
                size=(batchSize, self.hparams.latentSize),
                device=device)

        return self.raycast(z, phis, thetas)
