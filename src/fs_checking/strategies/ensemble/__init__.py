"""Ensemble strategy: 10x Flash detection + validation.

Recommended approach based on benchmarking:
- 10 parallel detection passes with shuffled page orders
- Validation pass to confirm and deduplicate
- 88.9% recall at $0.50 cost

Usage:
    uv run python -m fs_checking.strategies.ensemble "document.pdf"
"""

from .ensemble import run_ensemble

__all__ = ["run_ensemble"]
