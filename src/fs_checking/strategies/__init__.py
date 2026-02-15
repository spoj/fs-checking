"""Detection strategies for financial statement error checking."""

from .baseline import run_baseline
from .ensemble import run_ensemble
from .single_agent import run_single_agent
from .swarm import run_swarm

__all__ = ["run_baseline", "run_ensemble", "run_single_agent", "run_swarm"]
