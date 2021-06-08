import torch
import pyprof
from argparse import ArgumentParser
from dfr.__main__ import main, setArgs

pyprof.init(enable_function_stack=True)
torch.backends.cudnn.benchmark = True
parser = ArgumentParser()
setArgs(parser)
args = parser.parse_args()
args.no_log = True
args.profile = True

with torch.autograd.profiler.emit_nvtx():
    main(args)
