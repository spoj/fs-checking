"""Detection strategies for financial statement error checking."""

from .baseline import run_baseline
from .ensemble import run_ensemble
from .swarm import run_swarm

__all__ = ["run_baseline", "run_ensemble", "run_swarm"]
