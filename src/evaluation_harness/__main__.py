"""Entrypoint: `python -m evaluation_harness`."""

from __future__ import annotations

from evaluation_harness import __version__
from evaluation_harness.configuration import settings


def main() -> None:
    header = f"Evaluation Harness v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
