"""Repository-root entry point for CodeContext.

For users who installed via pip, the `codecontext` command is the preferred
entry point (defined in pyproject.toml). This `main.py` exists so that the
server and CLI also work when running directly from a git clone without
installing, matching the examples in PRODUCT_API.md:

    python main.py serve-api --root . --host 127.0.0.1 --port 8787
    python main.py metrics-report --window 7d

Both invocations call the exact same function, so behaviour is identical.
"""

from __future__ import annotations

import sys

from codecontext.cli import main


if __name__ == "__main__":
    sys.exit(main())
