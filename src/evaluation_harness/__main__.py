"""Enable ``python -m evaluation_harness`` to invoke the command-line interface."""

from __future__ import annotations

from evaluation_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
