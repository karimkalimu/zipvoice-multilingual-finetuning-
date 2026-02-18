#!/usr/bin/env python3
"""
Filter fbank cuts with NaN/Inf features or extreme token/frame ratios.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from lhotse import load_manifest


def _setup_logging(level: str) -> None:
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=getattr(logging, level), force=True)


def _tokens_len(cut) -> int:
    if not cut.supervisions:
        return 0
    sup = cut.supervisions[0]
    tokens = getattr(sup, "tokens", None)
    if isinstance(tokens, list):
        return len(tokens)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames-per-token", type=float, default=1.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cuts = load_manifest(args.input)
    total = len(cuts)
    logging.info("Loaded cuts: %d", total)

    kept = []
    dropped_nan = 0
    dropped_ratio = 0
    dropped_zero = 0
    for cut in cuts:
        try:
            feats = cut.load_features()
        except Exception:
            dropped_nan += 1
            continue

        if feats is None or feats.size == 0:
            dropped_zero += 1
            continue

        if not np.isfinite(feats).all():
            dropped_nan += 1
            continue

        frames = feats.shape[0]
        tok_len = _tokens_len(cut)
        if tok_len <= 0:
            dropped_zero += 1
            continue

        if frames / tok_len < args.min_frames_per_token:
            dropped_ratio += 1
            continue

        kept.append(cut)

    logging.info(
        "Kept: %d | Dropped: nan=%d ratio=%d zero=%d",
        len(kept),
        dropped_nan,
        dropped_ratio,
        dropped_zero,
    )

    from lhotse import CutSet

    CutSet.from_cuts(kept).to_file(args.output)
    logging.info("Saved filtered cuts to %s", args.output)


if __name__ == "__main__":
    main()
