import torch
import numpy as np
from tabulate import tabulate

values = {}

def log_memory(loc):
    global values
    count = torch.cuda.memory_allocated() / (1024 ** 2)
    if not loc in values:
        values[loc] = []

    values[loc].append(count)

def clear_memory_stats():
    global values
    values = {}

def print_memory_stats():
    global values
    statistics = []
    for key, val, in values.items():
        statistics.append([
            key,
            np.mean(val),
            np.max(val),
            np.min(val),
        ])

    print(tabulate(statistics, headers=['loc', 'mean', 'max', 'min']))
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))
