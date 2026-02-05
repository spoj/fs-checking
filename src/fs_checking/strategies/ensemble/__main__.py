"""Allow running as: python -m fs_checking.strategies.ensemble"""

import asyncio
from .ensemble import main

if __name__ == "__main__":
    asyncio.run(main())
