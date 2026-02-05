"""Swarm strategy: Multi-agent collaborative detection.

Experimental approach using multiple specialized agents.
More complex, higher cost, but potentially better coverage.

Usage:
    uv run python -m fs_checking.strategies.swarm "document.pdf"
"""

from .swarm import run_swarm

__all__ = ["run_swarm"]
