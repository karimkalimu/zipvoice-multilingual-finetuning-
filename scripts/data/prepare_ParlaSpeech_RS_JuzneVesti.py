# prepare_ParlaSpeech_RS_JuzneVesti.py

#!/usr/bin/env python3
"""
Prepare combined Juzne Vesti + ParlaSpeech-RS TSVs.

- Juzne Vesti JSONL is processed with speaker breakdowns and per-speaker
  segment WAVs (same logic as prepare_JuzneVesti.py).
- ParlaSpeech-RS JSON items are appended to the *train* TSV (same logic as
  prepare_ParlaSpeech_RS.py).
- JuzneVesti-SR-24khz-splitted will be created

python egs/zipvoice/local/prepare_ParlaSpeech_RS_JuzneVesti.py \
  --jv-audio-dir JuzneVesti-SR-24khz \
  --jv-jsonl-path JuzneVesti-SR/JuzneVesti-SR.v1.0.jsonl \
  --jv-split-audio-dir JuzneVesti-SR-24khz-splitted \
  --jv-path-strip-prefix seg_audio/ \
  --ps-audio-dir ParlaSpeech-RS \
  --ps-json-path ParlaSpeech-RS/ParlaSpeech-RS.v1.0_200_per_speaker_wav24k_mono.json \
  --output-dir data/raw \
  --log-level INFO
  

Outputs:
  data/raw/juznevesti_train.tsv  (Juzne Vesti train + ParlaSpeech-RS)
  data/raw/juznevesti_dev.tsv    (Juzne Vesti only)
  data/raw/juznevesti_test.tsv   (created but intentionally empty)
"""

from __future__ import annotations

import argparse
import audioop
import json
import logging
import re
import wave
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# === Juzne Vesti defaults ===
JUZNEVESTI_AUDIO_DIR = Path("JuzneVesti-SR-24khz")
JUZNEVESTI_JSONL_PATH = Path("JuzneVesti-SR/JuzneVesti-SR.v1.0.jsonl")
JUZNEVESTI_SPLIT_AUDIO_DIR = Path("JuzneVesti-SR-24khz-splitted")

# === ParlaSpeech-RS defaults ===
PARLASPEECH_AUDIO_DIR = Path("ParlaSpeech-RS")
PARLASPEECH_JSON_PATH = Path("ParlaSpeech-RS/ParlaSpeech-RS.v1.0_200_per_speaker_wav24k_mono.json")

# === Output directory ===
OUTPUT_DIR = Path("data/raw")

# === Juzne Vesti audio trimming ===
TRIM_DB = -48.0
TRIM_FRAME_MS = 10
MIN_SEGMENT_SEC = 0.2
TIME_TOLERANCE = 0.05


def _setup_logging(level: str) -> None:
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=getattr(logging, level), force=True)


def _make_unique_id(uid: str, seen_ids: set[str]) -> str:
    if uid not in seen_ids:
        seen_ids.add(uid)
        return uid
    i = 2
    while f"{uid}_{i}" in seen_ids:
        i += 1
    new_uid = f"{uid}_{i}"
    seen_ids.add(new_uid)
    return new_uid


# ---------------------- Juzne Vesti helpers ----------------------


def _jv_clean_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join([t for t in text if t])
    # Remove anchor tags and JV prefixes
    text = text.replace("<anchor_start>", " ").replace("<anchor_end>", " ")
    text = re.sub(r"\bJV\s*:\s*", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _jv_safe_id(orig_file: str, start: float, end: float) -> str:
    base = Path(orig_file).stem
    start_ms = int(round(start * 1000))
    end_ms = int(round(end * 1000))
    return f"{base}_{start_ms}_{end_ms}"


def _jv_safe_speaker_label(label: str) -> str:
    if not label:
        return "speaker"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip().lower()).strip("_")
    return safe or "speaker"


def _jv_resolve_wav_path(path_str: str, audio_dir: Path, jsonl_dir: Path, strip_prefixes: list[str]) -> Path | None:
    if not path_str:
        return None
    norm = path_str
    for prefix in strip_prefixes:
        if norm.startswith(prefix):
            norm = norm[len(prefix) :]
    p = Path(norm)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(jsonl_dir / p)
    candidates.append(audio_dir / p)

    for cand in candidates:
        if cand.exists():
            if cand != p:
                logging.debug("Resolved path %s -> %s", p, cand)
            return cand
    return None


def _read_wav_data(path: Path):
    try:
        with wave.open(str(path), "rb") as wf:
            params = wf.getparams()
            data = wf.readframes(params.nframes)
        return params, data
    except Exception:
        return None, b""


def _slice_audio_bytes(data: bytes, params, start_sec: float, end_sec: float) -> bytes:
    frame_size = params.sampwidth * params.nchannels
    if frame_size <= 0:
        return b""
    total_frames = len(data) // frame_size
    start_frame = int(round(start_sec * params.framerate))
    end_frame = int(round(end_sec * params.framerate))
    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames
    if end_frame <= start_frame:
        return b""
    return data[start_frame * frame_size : end_frame * frame_size]


def _db_to_rms(sampwidth: int, db: float) -> float:
    if sampwidth <= 0:
        return 0.0
    max_amp = float((1 << (8 * sampwidth - 1)) - 1)
    return max_amp * (10 ** (db / 20.0))


def _trim_silence(
    data: bytes,
    sample_rate: int,
    sampwidth: int,
    nchannels: int,
    threshold_db: float = TRIM_DB,
    frame_ms: int = TRIM_FRAME_MS,
):
    frame_size = sampwidth * nchannels
    if frame_size <= 0:
        return b""
    total_frames = len(data) // frame_size
    if total_frames == 0 or sample_rate <= 0:
        return b""
    win_frames = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    threshold_rms = _db_to_rms(sampwidth, threshold_db)

    lead_frame = None
    for i in range(0, total_frames, win_frames):
        end = min(total_frames, i + win_frames)
        chunk = data[i * frame_size : end * frame_size]
        if audioop.rms(chunk, sampwidth) >= threshold_rms:
            lead_frame = i
            break
    if lead_frame is None:
        return b""

    tail_frame = None
    for i in range(total_frames, 0, -win_frames):
        start = max(0, i - win_frames)
        chunk = data[start * frame_size : i * frame_size]
        if audioop.rms(chunk, sampwidth) >= threshold_rms:
            tail_frame = i
            break
    if tail_frame is None or tail_frame <= lead_frame:
        return b""

    return data[lead_frame * frame_size : tail_frame * frame_size]


def _write_wav_data(path: Path, params, data: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(data)


def _jv_get_timed_words(item):
    norm_words = item.get("norm_words")
    norm_starts = item.get("norm_words_start_times")
    if isinstance(norm_words, list) and isinstance(norm_starts, list) and len(norm_words) == len(norm_starts):
        return norm_words, norm_starts

    words = item.get("words")
    if isinstance(words, list) and words:
        if isinstance(words[0], dict):
            w_list = []
            t_list = []
            for w in words:
                if not isinstance(w, dict):
                    continue
                token = w.get("word") or w.get("text")
                ts = w.get("start") or w.get("start_time") or w.get("s")
                if token is None or ts is None:
                    continue
                try:
                    t_list.append(float(ts))
                    w_list.append(token)
                except Exception:
                    continue
            if w_list and len(w_list) == len(t_list):
                return w_list, t_list
    return None, None


def _jv_select_audio_source(item: dict, audio_dir: Path, jsonl_dir: Path, strip_prefixes: list[str]):
    path_str = item.get("path")
    seg_path = _jv_resolve_wav_path(path_str, audio_dir, jsonl_dir, strip_prefixes)
    if seg_path is None:
        return None, None, None, None, "missing"

    params, wav_data = _read_wav_data(seg_path)
    if params is None or params.framerate <= 0 or params.sampwidth <= 0:
        return None, None, None, None, "invalid_audio"

    wav_dur = params.nframes / float(params.framerate)
    return seg_path, params, wav_data, wav_dur, "ok"


def _prepare_juznevesti(args, out_files: dict[str, object], seen_ids: set[str]):
    if not args.jv_audio_dir.exists():
        raise FileNotFoundError(f"JuzneVesti AUDIO_DIR not found: {args.jv_audio_dir}")
    if not args.jv_jsonl_path.exists():
        raise FileNotFoundError(f"JuzneVesti JSONL not found: {args.jv_jsonl_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.jv_split_audio_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = args.jv_jsonl_path.parent

    total = 0
    kept = 0
    kept_by_split = {"train": 0, "dev": 0, "test": 0}
    missing_audio = 0
    missing_text = 0
    missing_time = 0
    missing_breakdown = 0
    missing_timed_words = 0
    bad_timed_words = 0
    bad_breakdown = 0
    bad_segment_duration = 0
    bad_path = 0
    bad_split = 0
    zero_duration = 0
    wav_duration_zero = 0
    trimmed_to_zero = 0
    too_short = 0

    with args.jv_jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing Juzne Vesti JSONL"):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            split = item.get("split", "train")
            if split == "test":
                split = "train"
            if split not in out_files:
                bad_split += 1
                continue

            path = item.get("path")
            if not path:
                missing_audio += 1
                continue

            wav_path, params, wav_data, wav_dur, status = _jv_select_audio_source(
                item,
                args.jv_audio_dir,
                jsonl_dir,
                args.jv_path_strip_prefix,
            )
            if status != "ok":
                bad_path += 1
                continue
            if params is None or params.framerate <= 0 or params.sampwidth <= 0:
                wav_duration_zero += 1
                continue
            frame_size = params.sampwidth * params.nchannels
            if frame_size <= 0:
                wav_duration_zero += 1
                continue
            if wav_dur <= 0:
                wav_duration_zero += 1
                continue

            start = item.get("start")
            end = item.get("end")
            try:
                start = float(start) if start is not None else None
                end = float(end) if end is not None else None
            except Exception:
                missing_time += 1
                continue
            if start is None or end is None:
                missing_time += 1
                continue
            seg_len = max(0.0, end - start)
            if abs(wav_dur - seg_len) > args.jv_seg_dur_tolerance:
                bad_segment_duration += 1
                logging.warning(
                    "Rejecting item due to segment duration mismatch: wav=%.2f seg=%.2f path=%s",
                    wav_dur,
                    seg_len,
                    path,
                )
                continue

            speaker_info = item.get("speaker_info") or {}
            breakdown = speaker_info.get("speaker_breakdown") or []
            if not breakdown:
                missing_breakdown += 1
                if args.jv_require_breakdown:
                    continue
                breakdown = [["unknown", 0.0, wav_dur]]

            timed_words, timed_starts = _jv_get_timed_words(item)
            if timed_words is None:
                missing_timed_words += 1
                if args.jv_require_timed_words:
                    continue

            breakdown_invalid = False
            for part in breakdown:
                if not isinstance(part, (list, tuple)) or len(part) < 3:
                    breakdown_invalid = True
                    break
                try:
                    p_start = float(part[1])
                    p_end = float(part[2])
                except Exception:
                    breakdown_invalid = True
                    break
                if p_start < 0.0 or p_end <= p_start or p_end > wav_dur + TIME_TOLERANCE:
                    breakdown_invalid = True
                    break
            if breakdown_invalid:
                bad_breakdown += 1
                continue

            if timed_starts:
                timed_invalid = False
                for ts in timed_starts:
                    try:
                        tsv = float(ts)
                    except Exception:
                        timed_invalid = True
                        break
                    if tsv < 0.0 or tsv > wav_dur + TIME_TOLERANCE:
                        timed_invalid = True
                        break
                if timed_invalid:
                    bad_timed_words += 1
                    continue

            for idx, part in enumerate(breakdown):
                if not isinstance(part, (list, tuple)) or len(part) < 3:
                    continue
                speaker = part[0]
                try:
                    part_start = float(part[1])
                    part_end = float(part[2])
                except Exception:
                    continue

                if part_start < 0.0:
                    part_start = 0.0
                if part_end > wav_dur:
                    part_end = wav_dur
                if part_end <= part_start:
                    zero_duration += 1
                    continue

                if timed_words is not None:
                    part_words = [w for w, ts in zip(timed_words, timed_starts) if ts >= part_start and ts < part_end]
                    text = _jv_clean_text(part_words)
                    if not text:
                        missing_text += 1
                        continue
                else:
                    if len(breakdown) > 1:
                        missing_text += 1
                        continue
                    text = _jv_clean_text(item.get("words") or item.get("norm_words") or "")
                    if not text:
                        missing_text += 1
                        continue

                part_data = _slice_audio_bytes(wav_data, params, part_start, part_end)
                if not part_data:
                    zero_duration += 1
                    continue

                trimmed_data = _trim_silence(
                    part_data,
                    params.framerate,
                    params.sampwidth,
                    params.nchannels,
                    threshold_db=args.jv_trim_db,
                    frame_ms=args.jv_trim_frame_ms,
                )
                if not trimmed_data:
                    trimmed_to_zero += 1
                    continue

                trimmed_frames = len(trimmed_data) // frame_size
                if trimmed_frames <= 0:
                    trimmed_to_zero += 1
                    continue

                end_out = trimmed_frames / float(params.framerate)
                if end_out < args.jv_min_segment_sec:
                    too_short += 1
                    continue

                abs_start = start + part_start
                abs_end = start + part_end
                uid_base = _jv_safe_id(item.get("orig_file", wav_path.name), abs_start, abs_end)
                speaker_tag = _jv_safe_speaker_label(speaker)
                uid_base = f"{uid_base}_{speaker_tag}"
                uid = uid_base
                out_path = args.jv_split_audio_dir / f"{uid}.wav"
                if out_path.exists() or uid in seen_ids:
                    suffix = 1
                    while True:
                        cand_uid = f"{uid_base}_{suffix}"
                        cand_path = args.jv_split_audio_dir / f"{cand_uid}.wav"
                        if cand_uid not in seen_ids and not cand_path.exists():
                            uid = cand_uid
                            out_path = cand_path
                            break
                        suffix += 1
                uid = _make_unique_id(uid, seen_ids)

                _write_wav_data(out_path, params, trimmed_data)

                out_files[split].write(f"{uid}\t{text}\t{out_path.as_posix()}\t0.0\t{end_out}\n")
                kept += 1
                kept_by_split[split] += 1

    return {
        "total": total,
        "kept": kept,
        "kept_by_split": kept_by_split,
        "missing_audio": missing_audio,
        "missing_text": missing_text,
        "missing_time": missing_time,
        "missing_breakdown": missing_breakdown,
        "missing_timed_words": missing_timed_words,
        "bad_timed_words": bad_timed_words,
        "bad_breakdown": bad_breakdown,
        "bad_segment_duration": bad_segment_duration,
        "bad_path": bad_path,
        "bad_split": bad_split,
        "zero_duration": zero_duration,
        "wav_duration_zero": wav_duration_zero,
        "trimmed_to_zero": trimmed_to_zero,
        "too_short": too_short,
    }


# ---------------------- ParlaSpeech-RS helpers ----------------------


def _ps_clean_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join([t for t in text if t])
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _ps_safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return safe.strip("_") or "utt"


def _ps_resolve_audio_path(path_str: str, audio_dir: Path, json_dir: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(str(path_str))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(json_dir / p)
    candidates.append(audio_dir / p)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _prepare_parlaspeech(args, out_train, seen_ids: set[str]):
    if not args.ps_json_path.exists():
        raise FileNotFoundError(f"ParlaSpeech-RS JSON not found: {args.ps_json_path}")
    if not args.ps_audio_dir.exists():
        raise FileNotFoundError(f"ParlaSpeech-RS AUDIO_DIR not found: {args.ps_audio_dir}")

    json_dir = args.ps_json_path.parent

    items = json.loads(args.ps_json_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("ParlaSpeech-RS input JSON must be a list of items")

    total = 0
    kept = 0
    missing_audio = 0
    missing_text = 0
    bad_split = 0

    dur_sum = 0.0
    dur_n = 0
    dur_min = None
    dur_max = None
    dur_missing = 0
    too_short = 0
    too_long = 0

    for item in tqdm(items, desc="Processing ParlaSpeech-RS JSON"):
        total += 1

        split = item.get("split", "train")
        if split != "train":
            bad_split += 1
            continue

        audio_rel = item.get("audio") or ""
        wav_path = _ps_resolve_audio_path(audio_rel, args.ps_audio_dir, json_dir)
        if wav_path is None:
            missing_audio += 1
            continue

        text = _ps_clean_text(item.get("text"))
        if not text:
            missing_text += 1
            continue

        base_id = item.get("id") or Path(str(audio_rel)).stem or wav_path.stem
        speaker = (item.get("speaker_info") or {}).get("Speaker_ID") or ""
        uid = _ps_safe_id(f"{base_id}_{speaker}") if speaker else _ps_safe_id(base_id)
        if args.ps_id_prefix:
            uid = _ps_safe_id(f"{args.ps_id_prefix}_{uid}")

        dur = None
        try:
            info = sf.info(str(wav_path))
            if info.samplerate and info.frames is not None:
                dur = float(info.frames) / float(info.samplerate)
        except Exception:
            pass
        if dur is None:
            try:
                dur = float(item.get("audio_length"))
            except Exception:
                dur = None

        if dur is None or dur <= 0:
            dur_missing += 1
            continue

        if dur < args.ps_min_segment_sec:
            too_short += 1
            continue
        if dur > args.ps_max_segment_sec:
            too_long += 1
            continue

        dur_sum += dur
        dur_n += 1
        dur_min = dur if dur_min is None else min(dur_min, dur)
        dur_max = dur if dur_max is None else max(dur_max, dur)

        uid = _make_unique_id(uid, seen_ids)
        out_train.write(f"{uid}\t{text}\t{wav_path.as_posix()}\t0.0\t{dur}\n")
        kept += 1

    return {
        "total": total,
        "kept": kept,
        "missing_audio": missing_audio,
        "missing_text": missing_text,
        "bad_split": bad_split,
        "dur_sum": dur_sum,
        "dur_n": dur_n,
        "dur_min": dur_min,
        "dur_max": dur_max,
        "dur_missing": dur_missing,
        "too_short": too_short,
        "too_long": too_long,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    # Juzne Vesti arguments
    parser.add_argument("--jv-audio-dir", type=Path, default=JUZNEVESTI_AUDIO_DIR)
    parser.add_argument("--jv-jsonl-path", type=Path, default=JUZNEVESTI_JSONL_PATH)
    parser.add_argument("--jv-split-audio-dir", type=Path, default=JUZNEVESTI_SPLIT_AUDIO_DIR)
    parser.add_argument("--jv-trim-db", type=float, default=TRIM_DB)
    parser.add_argument("--jv-trim-frame-ms", type=int, default=TRIM_FRAME_MS)
    parser.add_argument("--jv-min-segment-sec", type=float, default=MIN_SEGMENT_SEC)
    parser.add_argument("--jv-seg-dur-tolerance", type=float, default=0.2)
    parser.add_argument("--jv-require-breakdown", type=int, default=1)
    parser.add_argument("--jv-require-timed-words", type=int, default=0)
    parser.add_argument("--jv-path-strip-prefix", action="append", default=[])

    # ParlaSpeech-RS arguments
    parser.add_argument("--ps-audio-dir", type=Path, default=PARLASPEECH_AUDIO_DIR)
    parser.add_argument("--ps-json-path", type=Path, default=PARLASPEECH_JSON_PATH)
    parser.add_argument("--ps-min-segment-sec", type=float, default=0.5)
    parser.add_argument("--ps-max-segment-sec", type=float, default=30.0)
    parser.add_argument("--ps-id-prefix", type=str, default="ps")

    # Shared
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    _setup_logging(args.log_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "juznevesti_train.tsv"
    dev_path = args.output_dir / "juznevesti_dev.tsv"
    test_path = args.output_dir / "juznevesti_test.tsv"

    out_files = {
        "train": train_path.open("w", encoding="utf-8"),
        "dev": dev_path.open("w", encoding="utf-8"),
        "test": test_path.open("w", encoding="utf-8"),
    }

    seen_ids = set()
    try:
        jv_stats = _prepare_juznevesti(args, out_files, seen_ids)
        ps_stats = _prepare_parlaspeech(args, out_files["train"], seen_ids)
    finally:
        for fh in out_files.values():
            fh.close()

    print("=== Juzne Vesti stats ===")
    print(f"Total lines: {jv_stats['total']}")
    print(f"Kept: {jv_stats['kept']}")
    print(
        "Kept by split: "
        f"train={jv_stats['kept_by_split']['train']} "
        f"dev={jv_stats['kept_by_split']['dev']} "
        f"test={jv_stats['kept_by_split']['test']}"
    )
    print(f"Missing audio: {jv_stats['missing_audio']}")
    print(f"Missing text: {jv_stats['missing_text']}")
    print(f"Missing time: {jv_stats['missing_time']}")
    print(f"Missing speaker breakdown: {jv_stats['missing_breakdown']}")
    print(f"Missing timed words: {jv_stats['missing_timed_words']}")
    print(f"Bad timed words: {jv_stats['bad_timed_words']}")
    print(f"Bad breakdown: {jv_stats['bad_breakdown']}")
    print(f"Bad segment duration: {jv_stats['bad_segment_duration']}")
    print(f"Bad path: {jv_stats['bad_path']}")
    print(f"Bad split: {jv_stats['bad_split']}")
    print(f"Zero/neg durations: {jv_stats['zero_duration']}")
    print(f"Wav duration zero/invalid: {jv_stats['wav_duration_zero']}")
    print(f"Trimmed to zero: {jv_stats['trimmed_to_zero']}")
    print(f"Too short (<{args.jv_min_segment_sec:.1f}s): {jv_stats['too_short']}")

    print("=== ParlaSpeech-RS stats (appended to train) ===")
    print(f"Total items: {ps_stats['total']}")
    print(f"Kept: {ps_stats['kept']}")
    print(f"Missing audio: {ps_stats['missing_audio']}")
    print(f"Missing text: {ps_stats['missing_text']}")
    print(f"Bad split: {ps_stats['bad_split']}")

    if ps_stats["dur_n"]:
        print(
            "Duration stats (over kept with duration): "
            f"n={ps_stats['dur_n']} total_sec={ps_stats['dur_sum']:.3f} | "
            f"{ps_stats['dur_sum']/3600:.1f} hours mean_sec={ps_stats['dur_sum']/ps_stats['dur_n']:.3f} "
            f"min_sec={ps_stats['dur_min']:.3f} max_sec={ps_stats['dur_max']:.3f}"
        )
    else:
        print("Duration stats: n=0")
    print(f"Duration missing/invalid (kept items): {ps_stats['dur_missing']}")
    print(f"Too short (<{args.ps_min_segment_sec:.1f}s): {ps_stats['too_short']}")
    print(f"Too long (>{args.ps_max_segment_sec:.1f}s): {ps_stats['too_long']}")

    for split, path in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
        print(f"Output file ({split}): {path}")
        if not path.exists():
            print("  [missing]")
            continue
        with path.open("r", encoding="utf-8") as fh:
            line1 = fh.readline().rstrip("\n")
            line2 = fh.readline().rstrip("\n")
        if line1:
            print(f"  1: {line1}")
        else:
            print("  1: [empty]")
        if line2:
            print(f"  2: {line2}")
        else:
            print("  2: [empty]")


if __name__ == "__main__":
    main()
