import torch
import pyprof
from dfr.__main__ import main

pyprof.init(enable_function_stack=True)
torch.backends.cudnn.benchmark = True

with torch.autograd.profiler.emit_nvtx():
    main()
