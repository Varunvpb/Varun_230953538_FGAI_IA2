"""Evaluate selected checkpoints on validation and test splits."""

from __future__ import annotations

import argparse
import sys

from src.pipeline import run_evaluate
from src.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_evaluate(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
