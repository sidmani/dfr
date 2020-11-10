import torch

t1 = torch.tensor([5.0], requires_grad=True)
t1.retain_grad()
t1.backward(retain_graph=True)
print(t1.grad)
t2 = -t1
t2.retain_grad()
t2.backward()
print(t1.grad)
print(t2.grad)
