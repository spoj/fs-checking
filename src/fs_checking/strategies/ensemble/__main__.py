"""Allow running as: python -m fs_checking.strategies.ensemble"""

import asyncio
import sys


def _main() -> None:
    # Import here to avoid the double-import warning when running
    # python -m fs_checking.strategies.ensemble.ensemble
    from .ensemble import main

    asyncio.run(main())


if __name__ == "__main__":
    _main()
