#!/usr/bin/env python3
"""
Filter out cuts with empty token lists from a tokenized Lhotse manifest.
"""

import argparse
import logging
from pathlib import Path

from lhotse import load_manifest


def _setup_logging(level: str) -> None:
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=getattr(logging, level), force=True)


def _has_tokens(cut) -> bool:
    if not cut.supervisions:
        return False
    sup = cut.supervisions[0]
    if not hasattr(sup, "tokens"):
        return False
    tokens = sup.tokens
    return isinstance(tokens, list) and len(tokens) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if args.output.exists():
        logging.info("Output %s already exists; skipping.", args.output)
        return

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading cuts from %s", args.input)
    cuts = load_manifest(args.input)
    total = len(cuts)
    logging.info("Total cuts: %d", total)

    filtered = cuts.filter(_has_tokens)
    kept = len(filtered)
    logging.info("Kept cuts: %d (dropped %d)", kept, total - kept)

    logging.info("Saving filtered cuts to %s", args.output)
    filtered.to_file(args.output)
    logging.info("Done")


if __name__ == "__main__":
    main()
