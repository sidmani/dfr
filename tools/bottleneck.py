import torch
from argparse import ArgumentParser
from torch.autograd.profiler import profile
from dfr.__main__ import main, setArgs

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

if __name__ == "__main__":
    parser = ArgumentParser()
    setArgs(parser)
    parser.add_argument(
        '--profile-sort',
        dest='profile_sort',
        default='cuda_time_total',
        choices=[
            'cpu_time',
            'cuda_time',
            'cpu_time_total',
            'cuda_time_total',
            'cpu_memory_usage',
            'cuda_memory_usage',
            'self_cpu_memory_usage',
            'self_cuda_memory_usage',
            'count'
        ]
    )
    parser.add_argument(
        '--profile-out',
        dest='profile_out',
        default='prof',
        help='file to save profiler stats to'
    )
    parser.add_argument(
        '--profile-trace',
        dest='profile_trace',
        action='store_true',
        default=False,
        help='Save a chrome://tracing trace file.'
    )
    args = parser.parse_args()
    args.profile = True
    args.no_log = True

    print(f'Profiling (sort wrt {args.profile_sort})')
    with profile(use_cuda=True, profile_memory=True, with_stack=True) as prof:
        main(args)

    out = prof.key_averages(group_by_stack_n=10).table(row_limit=50, sort_by=args.profile_sort, top_level_events_only=True)
    print(f'Done profiling; writing stats to {args.profile_out}.')
    file = open(args.profile_out, 'w')
    file.write(out)
    file.close()

    if args.profile_trace:
        prof.export_chrome_trace('trace')
