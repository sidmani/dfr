import torch
from torch.optim import Adam
from dfr.optim import stepDiscriminator, stepGenerator
from dfr.sdfNetwork import SDFNetwork
from dfr.discriminator import Discriminator
from dfr.train import HParams

def test_stepGenerator():
    hp = HParams()
    gen = SDFNetwork(hp, device=None)
    dis = Discriminator()
    genOpt = Adam(gen.parameters(), hp.learningRate)

    batch = torch.ones(4, hp.imageSize, hp.imageSize)
    fake = gen.sampleGenerator(batch.shape[0])

    oldGenParam = next(gen.parameters()).clone()
    oldDisParam = next(dis.parameters()).clone()
    stepGenerator(fake, dis, genOpt)
    newGenParam = next(gen.parameters()).clone()
    newDisParam = next(dis.parameters()).clone()

    assert not torch.equal(oldGenParam, newGenParam)
    assert torch.equal(oldDisParam, newDisParam)
    assert next(gen.parameters()).grad is None
    assert next(dis.parameters()).grad is None

def test_stepDiscriminator():
    hp = HParams()
    gen = SDFNetwork(hp, device=None)
    dis = Discriminator()
    disOpt = Adam(dis.parameters(), hp.learningRate)

    assert next(gen.parameters()).grad is None

    batch = torch.ones(4, hp.imageSize, hp.imageSize)
    fake = gen.sampleGenerator(batch.shape[0])

    oldGenParam = next(gen.parameters()).clone()
    oldDisParam = next(dis.parameters()).clone()
    stepDiscriminator(fake, batch, dis, disOpt)
    newGenParam = next(gen.parameters()).clone()
    newDisParam = next(dis.parameters()).clone()

    assert torch.equal(oldGenParam, newGenParam)
    assert not torch.equal(oldDisParam, newDisParam)
    assert next(dis.parameters()).grad is None
    assert next(gen.parameters()).grad is None
