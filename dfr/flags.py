from dataclasses import dataclass

# Flags are stuff related to training that don't affect the network itself
@dataclass
class _Flags:
    AMP: bool = False
    silent: bool = False
    profile: bool = False

Flags = _Flags()
