import torch

# make vectors unit length
def normalize(vec, dim, keepLength=False, eps=0):
  length = vec.norm(dim=dim, keepdim=True)
  unit = vec / (length + eps)
  if keepLength:
    return unit, length
  return unit

# compute a gradient, possibly with AMP
def grad(inputs, outputs, gradScaler=None):
  if gradScaler is not None:
    outputs = gradScaler.scale(outputs)
  out = torch.autograd.grad(outputs=outputs,
                            inputs=inputs,
                            grad_outputs=torch.ones_like(outputs),
                            create_graph=True)[0]
  if gradScaler is not None:
    return out / gradScaler.get_scale()
  return out

# uniform random value in a range
def rand_range(rnge, shape, device):
  a, b = rnge
  return torch.rand(shape, device=device) * (b - a) + a

# iterate over count objects in batches, where count possibly not divisible by batch
def iterBatch(count, batch):
  steps = count // batch
  for i in range(steps + 1):
    if i == steps:
      if count % batch == 0:
        break
      batch = count % batch

    yield i * batch, batch
