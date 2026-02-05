"""Baseline strategy: Single LLM call for error detection.

Simple but limited - one pass, one model.
Good for quick checks, not comprehensive.

Usage:
    uv run python -m fs_checking.strategies.baseline "document.pdf"
"""

from .baseline import run_baseline

__all__ = ["run_baseline"]
