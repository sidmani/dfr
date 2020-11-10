import torch
import torch.nn as nn
import numpy as np
from .raycast.frustum import buildFrustum, enumerateRays, sphereToRect
from .raycast.shader import fastRayIntegral, shade
from .raycast.sample import sampleUniform, scaleRays
from .geometricInit import geometricInit

class SDFNetwork(nn.Module):
    def __init__(self, hparams, device, width=512):
        super().__init__()
        assert width > hparams.latentSize + 3
        self.device = device
        self.layers = nn.ModuleList([
            nn.Linear(hparams.latentSize, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width - (hparams.latentSize + 3)),
            # skip connection from input: latent + x (into 5th layer)
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, 1),
        ])

        # the frustum calculation has spherical symmetry, so can precompute it
        self.frustum = buildFrustum(hparams.fov, hparams.imageSize, device)

        # SAL geometric initialization
        geometricInit(self.layers)

        self.hparams = hparams
        if hparams.weightNorm:
            for i in range(8):
                self.layers[i] = nn.utils.weight_norm(self.layers[i])

        # DeepSDF uses ReLU, SALD uses Softplus
        self.activation = nn.ReLU()

    def forward(self, pts, latents):
        # first 4 layers only deal with latents
        r = latents
        for i in range(4):
            r = self.activation(self.layers[i](r))

        # skip connection
        # per SAL supplementary: divide by sqrt(2) to normalize
        sampleCount = pts.shape[0] // latents.shape[0]
        processed = torch.repeat_interleave(r, sampleCount, dim=0)

        # may be a way to create this in parent
        expandedLatents = torch.repeat_interleave(latents, sampleCount, dim=0)

        # TODO: layer idx 4 can be split into separate matrices and cached
        # expected gain ~7%
        r = torch.cat([pts, expandedLatents, processed], dim=1) / np.sqrt(2)
        for i in range(3):
            r = self.activation(self.layers[i + 4](r))

        return torch.tanh(self.layers[7](r))

    def raycast(self, latents, phis, thetas):
        # build a rotated frustum for each input angle
        rays = enumerateRays(phis, thetas, self.frustum.phiSpace, self.frustum.thetaSpace)

        # uniformly sample distances from the camera in the unit sphere
        # unsqueeze because we're using the same sample values for all objects
        samples = sampleUniform(
                self.frustum.near,
                self.frustum.far,
                self.hparams.raySamples,
                self.device).unsqueeze(0)

        # compute the sampling points for each ray that intersects the unit sphere
        cameraLoc = sphereToRect(phis, thetas, self.frustum.cameraD)
        targets = scaleRays(
                rays[:, self.frustum.mask],
                samples[:, self.frustum.mask],
                cameraLoc)

        # compute intersections for rays
        values = fastRayIntegral(latents, targets, self, 10e-10)

        # shape [px, px, channels]
        result = torch.ones(rays.shape[:3], device=self.device)
        result[:, self.frustum.mask] = shade(values)
        return result

    def sampleGenerator(self, batchSize, phi=np.pi / 6.0):
        # elevation angle: phi = pi/6
        phis = torch.ones(batchSize, device=self.device) * phi
        # azimuthal angle: 0 <= theta < 2pi
        thetas = torch.rand(batchSize, device=self.device) * (2.0 * np.pi)
        # latents with mean 0, variance 0.33
        z = torch.normal(
                mean=0.0,
                std=np.sqrt(0.33),
                size=(batchSize, self.hparams.latentSize),
                device=self.device)

        return self.raycast(z, phis, thetas)
