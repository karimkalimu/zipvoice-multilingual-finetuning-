# prepare_arabic_dataset.py

#!/usr/bin/env python3
"""
Prepare Arabic dataset TSVs for ZipVoice.

Rules:
- Common Voice provides train/dev/test only (caps: train 10k, dev 2k, test 2k).
- All other datasets (Human_3, MGB2) are used for training only,

Usage:
  python egs/zipvoice/local/prepare_arabic_dataset.py

maybe in the future:
https://huggingface.co/datasets/MohamedRashad/SADA22
https://huggingface.co/datasets/Thomcles/YodaLingua-Arabic
https://huggingface.co/datasets/MBZUAI/ClArTTS
https://docs.google.com/forms/d/e/1FAIpQLSfUHqQAFByPOqjB0t7INv3eXxUVPFUFUa00VCxwB39WNGINKQ/viewform

Outputs:
  data/raw/arabic_train.tsv
  data/raw/arabic_dev.tsv
  data/raw/arabic_test.tsv
"""

import csv
import logging
import wave
from pathlib import Path


# -----------------------------
# Constants (edit as needed)
# -----------------------------
LOG_LEVEL = "INFO"
REQUIRE_AUDIO = True
MIN_DURATION_SEC = 0.5

OUTPUT_DIR = Path("data/raw")
OUTPUT_TRAIN = OUTPUT_DIR / "arabic_train.tsv"
OUTPUT_DEV = OUTPUT_DIR / "arabic_dev.tsv"
OUTPUT_TEST = OUTPUT_DIR / "arabic_test.tsv"

# Common Voice (Arabic)
CV_DIR = Path("cv_arabic/cv-corpus-24.0-2025-12-05/ar")
CV_TSV_NAME = "full_data_accepted.tsv"
CV_AUDIO_SUBDIR = "clips"
CV_MAX_TRAIN = 30_000
CV_MAX_DEV = 2_000
CV_MAX_TEST = 0

# MBZUAI/ArVoice (Human_3)
HUMAN3_ROOT = Path("Human_3")
HUMAN3_TSVS = [
    HUMAN3_ROOT / "train.tsv",
    HUMAN3_ROOT / "test.tsv",
]

# MohamedRashad/mgb2-arabic
MGB2_TRAIN_TSV = Path("mgb2_arabic/mgb2_train.tsv")
MGB2_TSV = Path("mgb2_arabic/mgb2_test_validation.tsv")
MGB2_AUDIO_ROOT = Path("/disk1/ZipVoice")
MGB2_MAX_DEV_FROM_TEST_VALIDATION = 2_000


def _setup_logging() -> None:
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=getattr(logging, LOG_LEVEL), force=True)


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def _safe_id(prefix: str, primary: str | None, fallback_path: str | None) -> str:
    if primary:
        return f"{prefix}_{str(primary).strip()}"
    if fallback_path:
        return f"{prefix}_{Path(fallback_path).stem}"
    return f"{prefix}_unknown"


def _resolve_audio_path(path_str: str | None, root: Path | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute() or root is None:
        return path
    return root / path


def _require_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _write_row(writer, utt_id: str, text: str, audio_path: Path) -> None:
    writer.write(f"{utt_id}\t{text}\t{audio_path}\n")


def _wav_duration_sec(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError):
        return None


def _process_common_voice(train_w, dev_w, test_w):
    cv_tsv = CV_DIR / CV_TSV_NAME
    cv_audio = CV_DIR / CV_AUDIO_SUBDIR
    _require_exists(cv_tsv, "Common Voice TSV")
    _require_exists(cv_audio, "Common Voice audio dir")

    kept = {"train": 0, "dev": 0, "test": 0}
    skipped_missing = 0
    skipped_overflow = {"train": 0, "dev": 0, "test": 0}
    bad_split = 0
    total = 0

    with open(cv_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            split = (row.get("split") or "train").strip().lower()
            if split not in kept:
                bad_split += 1
                split = "train"

            text = _clean_text(row.get("sentence", ""))
            rel_path = row.get("path")
            if not text or not rel_path:
                skipped_missing += 1
                continue

            audio_path = cv_audio / rel_path
            if REQUIRE_AUDIO and not audio_path.exists():
                skipped_missing += 1
                continue

            # Only keep up to CV_MAX_DEV dev items; everything else goes to train.
            if split == "dev" and kept["dev"] < CV_MAX_DEV:
                target_split = "dev"
            else:
                target_split = "train"

            if target_split == "train" and kept["train"] >= CV_MAX_TRAIN:
                skipped_overflow["train"] += 1
                continue
            if target_split == "dev" and kept["dev"] >= CV_MAX_DEV:
                skipped_overflow["dev"] += 1
                continue
            if target_split == "test" and kept["test"] >= CV_MAX_TEST:
                skipped_overflow["test"] += 1
                continue

            utt_id = _safe_id("cv", row.get("sentence_id"), rel_path)
            if target_split == "train":
                _write_row(train_w, utt_id, text, audio_path)
            elif target_split == "dev":
                _write_row(dev_w, utt_id, text, audio_path)
            else:
                _write_row(test_w, utt_id, text, audio_path)
            kept[target_split] += 1

    logging.info("Common Voice: total=%d kept=train:%d dev:%d test:%d", total, kept["train"], kept["dev"], kept["test"])
    logging.info("Common Voice: missing text/audio=%d bad_split=%d", skipped_missing, bad_split)
    logging.info(
        "Common Voice: overflow skipped train=%d dev=%d test=%d",
        skipped_overflow["train"],
        skipped_overflow["dev"],
        skipped_overflow["test"],
    )


def _process_human3(train_w):
    for tsv_path in HUMAN3_TSVS:
        _require_exists(tsv_path, "Human_3 TSV")

    kept = 0
    skipped_missing = 0
    skipped_short = 0
    total = 0

    for tsv_path in HUMAN3_TSVS:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total += 1
                text = _clean_text(row.get("text", ""))
                rel_path = row.get("wav_path")
                if not text or not rel_path:
                    skipped_missing += 1
                    continue

                audio_path = _resolve_audio_path(rel_path, HUMAN3_ROOT)
                if audio_path is None:
                    skipped_missing += 1
                    continue
                if REQUIRE_AUDIO and not audio_path.exists():
                    skipped_missing += 1
                    continue

                duration = _wav_duration_sec(audio_path)
                if duration is None or duration <= MIN_DURATION_SEC:
                    skipped_short += 1
                    continue

                utt_id = _safe_id("human3", row.get("utt_id"), rel_path)
                _write_row(train_w, utt_id, text, audio_path)
                kept += 1

    logging.info(
        "Human_3: total=%d kept=%d missing text/audio=%d short/unknown=%d",
        total,
        kept,
        skipped_missing,
        skipped_short,
    )


def _process_mgb2(train_w, dev_w):
    _require_exists(MGB2_TRAIN_TSV, "MGB2 train TSV")
    _require_exists(MGB2_TSV, "MGB2 test/validation TSV")

    kept_train = 0
    kept_dev = 0
    skipped_missing = 0
    skipped_short = 0
    total = 0

    with open(MGB2_TRAIN_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            text = _clean_text(row.get("text", ""))
            rel_path = row.get("wav_path")
            if not text or not rel_path:
                skipped_missing += 1
                continue

            audio_path = _resolve_audio_path(rel_path, MGB2_AUDIO_ROOT)
            if audio_path is None:
                skipped_missing += 1
                continue
            if REQUIRE_AUDIO and not audio_path.exists():
                skipped_missing += 1
                continue

            duration = _wav_duration_sec(audio_path)
            if duration is None or duration <= MIN_DURATION_SEC:
                skipped_short += 1
                continue

            utt_id = _safe_id("mgb2", row.get("utt_id"), rel_path)
            _write_row(train_w, utt_id, text, audio_path)
            kept_train += 1

    with open(MGB2_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            text = _clean_text(row.get("text", ""))
            rel_path = row.get("wav_path")
            if not text or not rel_path:
                skipped_missing += 1
                continue

            audio_path = _resolve_audio_path(rel_path, MGB2_AUDIO_ROOT)
            if audio_path is None:
                skipped_missing += 1
                continue
            if REQUIRE_AUDIO and not audio_path.exists():
                skipped_missing += 1
                continue

            duration = _wav_duration_sec(audio_path)
            if duration is None or duration <= MIN_DURATION_SEC:
                skipped_short += 1
                continue

            utt_id = _safe_id("mgb2", row.get("utt_id"), rel_path)
            if kept_dev < MGB2_MAX_DEV_FROM_TEST_VALIDATION:
                _write_row(dev_w, utt_id, text, audio_path)
                kept_dev += 1
            else:
                _write_row(train_w, utt_id, text, audio_path)
                kept_train += 1

    logging.info(
        "MGB2: total=%d kept_train=%d kept_dev=%d missing text/audio=%d short/unknown=%d",
        total,
        kept_train,
        kept_dev,
        skipped_missing,
        skipped_short,
    )


def main() -> None:
    _setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as train_w, open(OUTPUT_DEV, "w", encoding="utf-8") as dev_w, open(
        OUTPUT_TEST, "w", encoding="utf-8"
    ) as test_w:
        _process_common_voice(train_w, dev_w, test_w)
        _process_human3(train_w)
        _process_mgb2(train_w, dev_w)

    logging.info("Outputs: %s, %s, %s", OUTPUT_TRAIN, OUTPUT_DEV, OUTPUT_TEST)


if __name__ == "__main__":
    main()
