import torch
from torch.autograd.profiler import profile
from torch.cuda.amp import GradScaler
from dfr.__main__ import main
from dfr.generator import Generator
from dfr.hparams import HParams
from dfr.raycast import MultiscaleFrustum
from dfr.sdfNetwork import SDFNetwork
from dfr.positional import createBasis
from tqdm import tqdm

# device = torch.device('cuda')
# hp = HParams()
# basis = createBasis(hp.positional, device)
# sdf = SDFNetwork(hp, basis)
# frustum = MultiscaleFrustum(hp.fov, hp.raycastSteps, device)
# gen = Generator(sdf, frustum, hp).to(device)
# scaler = GradScaler()

with profile(use_cuda=True, profile_memory=True, with_stack=True) as prof:
    main()
    # for i in tqdm(range(10)):
    #     gen.sample(10, device=device, gradScaler=scaler)


# Arguments:
#     sort_by (str, optional): Attribute used to sort entries. By default
#         they are printed in the same order as they were registered.
#         Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
#         ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
#         ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
#     top_level_events_only(bool, optional): Boolean flag to determine the
#         selection of events to display. If true, the profiler will only
#         display events at top level like top-level invocation of python
#         `lstm`, python `add` or other functions, nested events like low-level
#         cpu/cuda ops events are omitted for profiler result readability.

print(prof.key_averages(group_by_stack_n=10).table(row_limit=50, sort_by="cuda_time_total", top_level_events_only=True))

prof.export_chrome_trace('trace')
