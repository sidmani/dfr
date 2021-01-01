import torch

def tensor_stats(z, name):
    print(f'{name} mean: {torch.mean(z)}')
    print(f'{name} var: {torch.var(z)}')
    print(f'{name} max: {torch.max(z)}')
    print(f'{name} min: {torch.min(z)}')
