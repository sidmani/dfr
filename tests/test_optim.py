import torch
from torch.optim import Adam
from dfr.optim import stepDiscriminator, stepGenerator
from dfr.sdfNetwork import SDFNetwork
from dfr.discriminator import Discriminator
from dfr.generator import Generator
from dfr.raycast.frustum import Frustum
from dfr.checkpoint import HParams, loadModel

def test_stepGenerator():
    models, optim, hp, _ = loadModel(None, None)
    gen, dis = models
    genOpt, _ = optim

    fake, normals = gen.sample(4)

    oldGenParam = next(gen.parameters()).clone()
    oldDisParam = next(dis.parameters()).clone()
    stepGenerator(fake, normals, dis, genOpt, 0.1)
    newGenParam = next(gen.parameters()).clone()
    newDisParam = next(dis.parameters()).clone()

    assert not torch.equal(oldGenParam, newGenParam)
    assert torch.equal(oldDisParam, newDisParam)
    assert next(gen.parameters()).grad is None
    assert next(dis.parameters()).grad is None

def test_stepDiscriminator():
    models, optim, hp, _ = loadModel(None, None)
    gen, dis = models
    _, disOpt = optim

    assert next(gen.parameters()).grad is None

    batch = torch.ones(5, 4, hp.imageSize, hp.imageSize)
    fake, normals = gen.sample(batch.shape[0])

    oldGenParam = next(gen.parameters()).clone()
    oldDisParam = next(dis.parameters()).clone()
    stepDiscriminator(fake, batch, dis, disOpt)
    newGenParam = next(gen.parameters()).clone()
    newDisParam = next(dis.parameters()).clone()

    assert torch.equal(oldGenParam, newGenParam)
    assert not torch.equal(oldDisParam, newDisParam)
    assert next(dis.parameters()).grad is None
    assert next(gen.parameters()).grad is None
