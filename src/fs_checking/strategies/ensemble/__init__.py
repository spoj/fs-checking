"""Ensemble strategy: Nx Flash detection + Pro rank/dedupe."""


def run_ensemble(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy-import wrapper to avoid runpy double-import warning."""
    from .ensemble import run_ensemble as _run

    return _run(*args, **kwargs)


__all__ = ["run_ensemble"]
