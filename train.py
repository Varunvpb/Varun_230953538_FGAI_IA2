"""Train all four recurrent models."""

from __future__ import annotations

import argparse
import sys

from src.pipeline import run_train
from src.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_train(load_config(args.config), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
