"""Generate temperature-sampled continuations for all models."""

from __future__ import annotations

import argparse
import sys

from src.pipeline import run_generate, run_visualize, write_reports
from src.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_generate(config)
    if not args.skip_reports:
        run_visualize(config)
        write_reports(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
