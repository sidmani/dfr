import pytorch_lightning as pl
from .sdfNetwork import SDFNetwork
from .raycast.frustum import buildFrustum
from .raycast import raycast

class Generator(pl.LightningModule):
    def __init__(self, weightNorm, fov, px, sampleCount, latentSize):
        super().__init__()
        self.sampleCount = sampleCount
        self.sdf = SDFNetwork(weightNorm=weightNorm, latentSize=latentSize)
        # the frustum calculation has spherical symmetry, so can precompute it
        self.frustum = buildFrustum(fov, px, self.device)

    def forward(self, latents, phis, thetas):
        return raycast(
                self.sdf,
                latents,
                phis,
                thetas,
                self.frustum,
                self.sampleCount,
                self.device)
