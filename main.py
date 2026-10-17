"""CLI for the Tiny Shakespeare word-level generation assignment."""

from __future__ import annotations

import argparse
import sys
import traceback

from src.pipeline import run_all, run_evaluate, run_generate, run_preprocess, run_train, run_visualize, write_reports
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICT4403 Assignment 2 pipeline")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "evaluate", "generate", "visualize", "report", "all"],
        default="all",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.max_vocab_size is not None:
        config["preprocessing"]["max_vocab_size"] = args.max_vocab_size
    if args.sequence_length is not None:
        config["preprocessing"]["sequence_length"] = args.sequence_length
    if args.window_stride is not None:
        config["preprocessing"]["window_stride"] = args.window_stride
    if args.force_retrain:
        config["training"]["force_retrain"] = True
    return config


def main() -> int:
    args = parse_args()
    try:
        config = apply_overrides(load_config(args.config), args)
        if args.mode == "preprocess":
            run_preprocess(config)
        elif args.mode == "train":
            run_train(config, force=args.force_retrain)
        elif args.mode == "evaluate":
            run_evaluate(config)
        elif args.mode == "generate":
            run_generate(config)
        elif args.mode == "visualize":
            run_visualize(config)
        elif args.mode == "report":
            write_reports(config)
        elif args.mode == "all":
            run_all(config, force_retrain=args.force_retrain)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
