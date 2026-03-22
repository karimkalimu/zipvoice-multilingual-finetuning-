#!/usr/bin/env python3
"""
Evaluate local ZipVoice checkpoints on TSV datasets prepared by:
  - egs/zipvoice/local/prepare_arabic_dataset.py
  - egs/zipvoice/local/prepare_ParlaSpeech_RS_JuzneVesti.py

 python evaluate_model.py \
  --dataset-tsv data/raw/arabic_dev.tsv \
  --lang ar \
  --checkpoint-path exp/zipvoice_ar/epoch-26-avg-4.pt \
  --model-config-path exp/pretrained_zipvoice/model.json \
  --token-file data/tokens_arabic_espeak_360.txt \
  --model-type zipvoice \
  --tokenizer espeak \
  --espeak-lang ar \
  --asr-language ar \
  --number-sentences 100 \
  --sim-checkpoint ./wavlm_large_finetune.pth \
  --sim-wavlm-ckpt ./wavlm_large.pt \
  --out-csv eval_outputs/arabic_eval.csv \
  --out-json eval_outputs/arabic_eval.json


 python evaluate_model.py \
  --dataset-tsv data/raw/juznevesti_dev.tsv \
  --lang sr \
  --checkpoint-path exp/zipvoice_sr/epoch-15-avg-4.safetensors \
  --model-dir exp/zipvoice_sr \
  --model-type zipvoice \
  --tokenizer espeak \
  --espeak-lang sr \
  --asr-language hr \
  --number-sentences 100 \
  --sim-checkpoint ./wavlm_large_finetune.pth \
  --sim-wavlm-ckpt ./wavlm_large.pt \
  --out-csv eval_outputs/serbian_eval.csv \
  --out-json eval_outputs/serbian_eval.json



  --model-type zipvoice_distill \


"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import safetensors.torch
import torch
import torchaudio
from jiwer import cer, wer
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


if "zipvoice" not in sys.modules:
    _zipvoice_master = Path(__file__).resolve().parent / "ZipVoice-master"
    _zipvoice_master_str = str(_zipvoice_master)
    if _zipvoice_master.is_dir() and _zipvoice_master_str not in sys.path:
        sys.path.insert(0, _zipvoice_master_str)

from zipvoice.bin.infer_zipvoice import VocosFbank, generate_sentence, get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint


@dataclass
class Sample:
    idx: int
    utt_id: str
    lang: str
    text: str
    wav_path: Path
    start: float | None
    end: float | None


def normalize_lang(lang: str) -> str:
    value = (lang or "").strip().lower()
    if value.startswith(("ar", "ara", "arabic")):
        return "ar"
    if value.startswith(("sr", "srp", "serbian")):
        return "sr"
    return value


def norm_text(text: str, lang: str) -> str:
    out = (text or "").strip()
    out = re.sub(r"\s+", " ", out)
    if normalize_lang(lang) == "ar":
        out = re.sub(r"[ًٌٍَُِّْـ]", "", out)
    out = re.sub(r"[^\w\s]", " ", out, flags=re.UNICODE)
    out = re.sub(r"\s+", " ", out).strip().casefold()
    return out


def collapse_repeated_patterns(text: str) -> str:
    tokens = (text or "").split()
    if not tokens:
        return ""
    dedup_tokens = [tokens[0]]
    for tok in tokens[1:]:
        if tok != dedup_tokens[-1]:
            dedup_tokens.append(tok)
    tokens = dedup_tokens
    n = len(tokens)
    for unit in range(1, (n // 2) + 1):
        if n % unit != 0:
            continue
        base = tokens[:unit]
        if base * (n // unit) == tokens:
            return " ".join(base)
    return " ".join(tokens)


def whisper_lang(lang: str, asr_language: str = "auto") -> str | None:
    forced = (asr_language or "auto").strip().lower()
    if forced != "auto":
        return forced
    value = normalize_lang(lang)
    if value == "ar":
        return "ar"
    if value == "sr":
        # Use Croatian decoding for Serbian to prefer Latin script output.
        return "hr"
    return None


def safe_wer(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return float(wer(ref, hyp))


def safe_cer(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return float(cer(ref, hyp))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset-tsv", type=Path, required=True, help="Input TSV with 3 columns (utt_id,text,wav_path) or 5 columns (utt_id,text,wav_path,start,end).")
    ap.add_argument("--lang", type=str, required=True, help="Dataset language (ar/arabic/sr/serbian).")
    ap.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N rows.")
    ap.add_argument(
        "--number-sentences",
        type=int,
        default=50,
        help="Select N samples with prompt durations closest to 1-3s. Use <=0 to keep all loaded rows.",
    )
    ap.add_argument("--prompt-mode", choices=["self", "fixed"], default="self", help="'self': use each row wav+text as prompt. 'fixed': use one fixed prompt for all rows.")
    ap.add_argument("--prompt-wav", type=Path, default=None, help="Required when --prompt-mode fixed.")
    ap.add_argument("--prompt-text", type=str, default=None, help="Required when --prompt-mode fixed.")

    ap.add_argument("--model-dir", type=Path, default=None, help="Optional model directory containing model.json and tokens.txt.")
    ap.add_argument("--checkpoint-path", type=Path, required=True, help="Path to checkpoint (.pt or .safetensors).")
    ap.add_argument("--model-config-path", type=Path, default=None, help="Path to model.json. If omitted, uses --model-dir/model.json.")
    ap.add_argument("--token-file", type=Path, default=None, help="Path to tokens.txt. If omitted, uses --model-dir/tokens.txt.")
    ap.add_argument("--model-type", choices=["auto", "zipvoice", "zipvoice_distill"], default="auto", help="Use auto to infer distill vs non-distill from paths.")
    ap.add_argument("--tokenizer", choices=["espeak", "emilia", "libritts", "simple"], default="espeak", help="Tokenizer type used by the checkpoint.")
    ap.add_argument("--espeak-lang", type=str, default=None, help="espeak language code; if omitted uses --lang.")
    ap.add_argument("--vocoder-path", type=Path, default=None, help="Optional local vocoder dir (config.yaml + pytorch_model.bin).")

    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-step", type=int, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--t-shift", type=float, default=0.5)
    ap.add_argument("--target-rms", type=float, default=0.1)
    ap.add_argument("--feat-scale", type=float, default=0.1)
    ap.add_argument("--max-duration", type=float, default=100.0)
    ap.add_argument(
        "--min-target-seconds",
        type=float,
        default=2.0,
        help="Minimum allowed target/reference duration for selected target rows.",
    )
    ap.add_argument(
        "--max-target-seconds",
        type=float,
        default=5.0,
        help="Maximum allowed target/reference duration for selected target rows; also caps generation max_duration.",
    )
    ap.add_argument("--remove-long-sil", action="store_true")

    ap.add_argument("--asr-model-id", type=str, default="openai/whisper-large-v3-turbo")
    ap.add_argument(
        "--asr-language",
        type=str,
        default="auto",
        help="Whisper decode language code. 'auto' uses ar->ar and sr->hr (Latin-friendly Serbian).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--hf-auth-token", type=str, default=None, help="Optional HuggingFace token for loading ASR/model assets.")
    ap.add_argument(
        "--sim-checkpoint",
        type=Path,
        default=Path("wavlm_large_finetune.pth"),
        help="Path to WavLM speaker verification checkpoint used by seed-tts-eval cal_sim.sh flow.",
    )
    ap.add_argument(
        "--sim-wavlm-ckpt",
        type=Path,
        default=None,
        help="Local upstream wavlm_large.pt (required for offline wavlm_large SIM to avoid HuggingFace download).",
    )
    ap.add_argument("--sim-model-name", type=str, default="wavlm_large", help="Speaker verification model name (seed-tts-eval).")
    ap.add_argument("--skip-sim", action="store_true", help="Skip speaker similarity evaluation.")

    ap.add_argument("--generated-dir", type=Path, default=Path("eval_outputs/generated_wavs"))
    ap.add_argument("--out-csv", type=Path, default=Path("eval_outputs/eval_results.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("eval_outputs/eval_results.json"))
    ap.add_argument("--show-worst", type=int, default=10, help="How many worst-WER samples to print.")
    ap.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap


def resolve_device(device_arg: str) -> torch.device:
    value = (device_arg or "auto").strip().lower()
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available in PyTorch.")
        try:
            cuda_idx = int(value.split(":", 1)[1])
        except ValueError as e:
            raise ValueError(f"Invalid CUDA device format: {device_arg}") from e
        return torch.device("cuda", cuda_idx)
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda", 0)
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def infer_model_type(model_type: str, checkpoint_path: Path, model_config_path: Path, model_dir: Path | None) -> str:
    if model_type != "auto":
        return model_type
    haystack = " ".join(
        [
            str(checkpoint_path).lower(),
            str(model_config_path).lower(),
            str(model_dir).lower() if model_dir is not None else "",
        ]
    )
    return "zipvoice_distill" if "distill" in haystack else "zipvoice"


def resolve_model_paths(args) -> tuple[Path, Path, Path]:
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None

    if args.model_config_path:
        model_config_path = args.model_config_path.expanduser().resolve()
    elif model_dir:
        model_config_path = (model_dir / "model.json").resolve()
    else:
        raise ValueError("Provide --model-config-path or --model-dir.")

    if args.token_file:
        token_file = args.token_file.expanduser().resolve()
    elif model_dir:
        token_file = (model_dir / "tokens.txt").resolve()
    else:
        raise ValueError("Provide --token-file or --model-dir.")

    for p in [checkpoint_path, model_config_path, token_file]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")
    return checkpoint_path, model_config_path, token_file


def load_samples(dataset_tsv: Path, lang: str, max_samples: int | None = None) -> list[Sample]:
    dataset_tsv = dataset_tsv.expanduser().resolve()
    if not dataset_tsv.is_file():
        raise FileNotFoundError(f"TSV not found: {dataset_tsv}")

    base_lang = normalize_lang(lang)
    samples: list[Sample] = []
    with dataset_tsv.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) == 3:
                utt_id, text, wav = cols
                start, end = None, None
            elif len(cols) >= 5:
                utt_id, text, wav, start_s, end_s = cols[:5]
                start = float(start_s)
                end = float(end_s)
            else:
                raise ValueError(f"Invalid TSV row at line {i + 1}: expected 3 or 5 columns, got {len(cols)}")

            wav_path = Path(wav).expanduser()
            if not wav_path.is_absolute():
                wav_path = (dataset_tsv.parent / wav_path).resolve()
                if not wav_path.exists():
                    wav_path = Path(wav).expanduser().resolve()
            if not wav_path.is_file():
                raise FileNotFoundError(f"Missing wav for line {i + 1}: {wav_path}")

            samples.append(Sample(idx=i, utt_id=utt_id, lang=base_lang, text=text, wav_path=wav_path, start=start, end=end))
            if max_samples is not None and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"No valid rows found in {dataset_tsv}")
    return samples


def prompt_duration_seconds(sample: Sample) -> float:
    if sample.start is not None and sample.end is not None:
        duration = float(sample.end) - float(sample.start)
        if duration > 0.0:
            return duration
    path_str = str(sample.wav_path)

    # Fast path for formats supported by current torchaudio backend.
    try:
        info = torchaudio.info(path_str)
        if int(info.sample_rate) > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    # Fallback to ffprobe (commonly available and robust for mp3).
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path_str,
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="ignore").strip()
        duration = float(out)
        if duration > 0.0:
            return duration
    except Exception:
        pass

    # Last fallback: librosa via audioread/soundfile.
    try:
        import librosa  # type: ignore

        duration = float(librosa.get_duration(path=path_str))
        if duration > 0.0:
            return duration
    except Exception:
        pass

    raise ValueError(
        f"Could not determine audio duration for {sample.wav_path}. "
        "Install ffmpeg/ffprobe or ensure torchaudio backend can decode this format."
    )


def select_samples_by_duration(samples: list[Sample], number_sentences: int, target_min: float = 2.0, target_max: float = 4.0) -> tuple[list[Sample], dict[int, float], int]:
    duration_by_idx: dict[int, float] = {}
    scored: list[tuple[float, float, int, Sample]] = []
    in_target_count = 0
    target_mid = 0.5 * (target_min + target_max)

    for sample in samples:
        duration = prompt_duration_seconds(sample)
        duration_by_idx[sample.idx] = duration
        if target_min <= duration <= target_max:
            distance = 0.0
            in_target_count += 1
        elif duration < target_min:
            distance = target_min - duration
        else:
            distance = duration - target_max
        scored.append((distance, abs(duration - target_mid), sample.idx, sample))

    if number_sentences <= 0 or number_sentences >= len(samples):
        return samples, duration_by_idx, in_target_count

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    selected = [item[3] for item in scored[:number_sentences]]
    return selected, duration_by_idx, in_target_count


def select_other_targets(
    samples: list[Sample],
    excluded_idx: set[int],
    number_sentences: int,
    seed: int,
    min_target_seconds: float,
    max_target_seconds: float,
    duration_by_idx: dict[int, float],
) -> tuple[list[Sample], int]:
    remaining = [s for s in samples if s.idx not in excluded_idx]
    if max_target_seconds > 0:
        remaining = [
            s
            for s in remaining
            if min_target_seconds
            <= float(duration_by_idx.get(s.idx, prompt_duration_seconds(s)))
            <= max_target_seconds
        ]
    eligible_count = len(remaining)
    if not remaining:
        return [], eligible_count
    if number_sentences <= 0 or number_sentences >= len(remaining):
        return remaining, eligible_count
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(remaining), size=number_sentences, replace=False).tolist())
    return [remaining[i] for i in chosen], eligible_count


def make_tokenizer(tokenizer_name: str, token_file: Path, espeak_lang: str):
    if tokenizer_name == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    if tokenizer_name == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    if tokenizer_name == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=espeak_lang)
    return SimpleTokenizer(token_file=token_file)


def load_tts_bundle(args, checkpoint_path: Path, model_config_path: Path, token_file: Path, runtime_device: torch.device):
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    espeak_lang = (args.espeak_lang or normalize_lang(args.lang) or "en").strip().lower()
    tokenizer = make_tokenizer(args.tokenizer, token_file, espeak_lang)
    model_type = infer_model_type(args.model_type, checkpoint_path, model_config_path, args.model_dir)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    if model_type == "zipvoice":
        model = ZipVoice(**model_config["model"], **tokenizer_config)
    else:
        model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)

    if str(checkpoint_path).endswith(".safetensors"):
        safetensors.torch.load_model(model, checkpoint_path)
    elif str(checkpoint_path).endswith(".pt"):
        load_checkpoint(filename=checkpoint_path, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported checkpoint format: {checkpoint_path}")

    device = runtime_device
    model = model.to(device).eval()
    vocoder = get_vocoder(str(args.vocoder_path) if args.vocoder_path else None)
    vocoder = vocoder.to(device).eval()

    if model_config["feature"]["type"] != "vocos":
        raise NotImplementedError(f"Unsupported feature type: {model_config['feature']['type']}")

    defaults = {"zipvoice": {"num_step": 16, "guidance_scale": 1.0}, "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0}}
    if args.num_step is None:
        args.num_step = defaults[model_type]["num_step"]
    if args.guidance_scale is None:
        args.guidance_scale = defaults[model_type]["guidance_scale"]

    return {
        "model": model,
        "tokenizer": tokenizer,
        "feature_extractor": VocosFbank(),
        "vocoder": vocoder,
        "sampling_rate": int(model_config["feature"]["sampling_rate"]),
        "device": device,
        "model_type": model_type,
        "espeak_lang": espeak_lang,
    }


def make_asr(args, runtime_device: torch.device):
    use_cuda = runtime_device.type == "cuda"
    pipe_device = int(runtime_device.index if runtime_device.index is not None else 0) if use_cuda else -1
    torch_dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        token=args.hf_auth_token,
    )
    processor = AutoProcessor.from_pretrained(args.asr_model_id, token=args.hf_auth_token)
    if use_cuda:
        model = model.to(runtime_device)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=pipe_device,
    )


@dataclass
class SimilarityScorer:
    verification_fn: Any
    model_name: str
    checkpoint_path: Path
    use_gpu: bool
    device: int | str
    model: Any = None

    def score(self, wav_a: Path, wav_b: Path) -> float:
        sim, self.model = self.verification_fn(
            self.model_name,
            str(wav_a),
            str(wav_b),
            use_gpu=self.use_gpu,
            checkpoint=str(self.checkpoint_path),
            wav1_start_sr=0,
            wav2_start_sr=0,
            wav1_end_sr=-1,
            wav2_end_sr=-1,
            model=self.model,
            device=self.device,
        )
        return float(sim.detach().cpu().item())


def make_similarity_scorer(args, runtime_device: torch.device) -> SimilarityScorer | None:
    if args.skip_sim:
        return None

    workdir = Path(__file__).resolve().parent
    verification_dir = workdir / "seed-tts-eval-main" / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"
    if not verification_dir.is_dir():
        raise FileNotFoundError(f"seed-tts-eval speaker_verification dir not found: {verification_dir}")

    checkpoint_path = args.sim_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Similarity checkpoint not found: {checkpoint_path}")

    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    verification_module = importlib.import_module("verification")
    verification_fn = getattr(verification_module, "verification")
    original_init_model = getattr(verification_module, "init_model")

    if args.sim_model_name == "wavlm_large":
        # seed-tts verification tries to download wavlm_large.pt when config_path=None.
        # In offline environments, force local upstream checkpoint usage.
        upstream_candidates: list[Path] = []
        if args.sim_wavlm_ckpt is not None:
            upstream_candidates.append(args.sim_wavlm_ckpt.expanduser())
        upstream_candidates.extend(
            [
                Path("wavlm_large.pt"),
                Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "wavlm_large.pt",
            ]
        )
        upstream_ckpt = None
        for cand in upstream_candidates:
            cand_abs = cand.resolve()
            if cand_abs.is_file():
                upstream_ckpt = cand_abs
                break

        if upstream_ckpt is None:
            raise FileNotFoundError(
                "Offline SIM setup missing upstream WavLM checkpoint. "
                "Provide --sim-wavlm-ckpt /path/to/wavlm_large.pt "
                "(wavlm_large_finetune.pth alone is not sufficient)."
            )
        if "finetune" in upstream_ckpt.name.lower():
            raise ValueError(
                f"--sim-wavlm-ckpt points to '{upstream_ckpt.name}', which is not the upstream backbone. "
                "Use wavlm_large.pt (not wavlm_large_finetune.pth)."
            )

        ecapa_module = importlib.import_module("models.ecapa_tdnn")
        if not hasattr(ecapa_module, "UpstreamExpert"):
            hub_root = Path(torch.hub.get_dir())
            hub_candidates = sorted(
                [p for p in hub_root.glob("s3prl_s3prl_*") if (p / "hubconf.py").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not hub_candidates:
                raise FileNotFoundError(
                    "Local s3prl torch.hub repo not found under "
                    f"{hub_root}. Run once with internet to cache s3prl, or provide a local s3prl repo."
                )
            s3prl_hub_dir = hub_candidates[0]

            class _LocalUpstreamExpert(torch.nn.Module):
                def __init__(self, ckpt_path: str):
                    super().__init__()
                    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
                    self.inner = torch.hub.load(
                        str(s3prl_hub_dir),
                        "wavlm_local",
                        source="local",
                        ckpt=ckpt_path,
                    )
                    self.model = self.inner.model

                def forward(self, wavs):
                    return self.inner(wavs)

            setattr(ecapa_module, "UpstreamExpert", _LocalUpstreamExpert)

        def _init_model_offline(model_name: str, checkpoint: str | None = None):
            if model_name != "wavlm_large":
                return original_init_model(model_name, checkpoint)
            model = ecapa_module.ECAPA_TDNN_SMALL(
                feat_dim=1024,
                feat_type="wavlm_large",
                config_path=str(upstream_ckpt),
            )
            if checkpoint is not None:
                state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
                model.load_state_dict(state_dict["model"], strict=False)
            return model

        verification_module.init_model = _init_model_offline
        setattr(args, "_sim_upstream_ckpt_used", str(upstream_ckpt))

    use_gpu = runtime_device.type == "cuda"
    device = int(runtime_device.index if runtime_device.index is not None else 0) if use_gpu else "cpu"
    return SimilarityScorer(
        verification_fn=verification_fn,
        model_name=args.sim_model_name,
        checkpoint_path=checkpoint_path,
        use_gpu=use_gpu,
        device=device,
    )


def safe_id(text: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip()).strip("_")
    return value or fallback


def load_audio_segment(path: Path, start: float | None = None, end: float | None = None) -> tuple[torch.Tensor, int]:
    path_str = str(path)

    # Try torchaudio first (fast path).
    try:
        if start is None and end is None:
            wav, sr = torchaudio.load(path_str)
            return wav, int(sr)

        info = torchaudio.info(path_str)
        sr = int(info.sample_rate)
        total_frames = int(info.num_frames)
        frame_start = max(0, int(round(max(0.0, float(start or 0.0)) * sr)))
        if end is None:
            frame_end = total_frames
        else:
            frame_end = min(total_frames, int(round(float(end) * sr)))
        frame_count = frame_end - frame_start
        if frame_count <= 0:
            raise ValueError(f"Invalid segment: start={start}, end={end}, path={path}")
        wav, sr_loaded = torchaudio.load(path_str, frame_offset=frame_start, num_frames=frame_count)
        return wav, int(sr_loaded)
    except Exception:
        pass

    # Fallback for mp3/unsupported codecs in torchaudio backend.
    try:
        import librosa  # type: ignore

        offset = max(0.0, float(start or 0.0))
        duration = None if end is None else max(0.0, float(end) - offset)
        y, sr = librosa.load(path_str, sr=None, mono=False, offset=offset, duration=duration)
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y = y[None, :]
        wav = torch.from_numpy(np.asarray(y)).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.numel() == 0:
            raise ValueError(f"Decoded empty waveform from {path}")
        return wav, int(sr)
    except Exception as e:
        raise ValueError(f"Failed to decode audio for prompt from {path}: {e}") from e


def ensure_prompt_wav(sample: Sample, prompt_dir: Path) -> Path:
    need_cache = sample.start is not None or sample.end is not None or sample.wav_path.suffix.lower() != ".wav"
    if not need_cache:
        return sample.wav_path

    out_name = safe_id(sample.utt_id, f"sample_{sample.idx:06d}") + "_prompt.wav"
    out_path = prompt_dir / out_name
    if out_path.exists():
        return out_path
    wav, sr = load_audio_segment(sample.wav_path, start=sample.start, end=sample.end)
    torchaudio.save(str(out_path), wav, sample_rate=sr)
    return out_path


def to_plain(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return v


def metric_summary(values: pd.Series) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def trunc_text(value: Any, max_len: int = 60) -> str:
    text = str(value if value is not None else "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def worst_reason(row: pd.Series) -> str:
    def as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    reasons: list[str] = []
    wer_v = as_float(row.get("wer", np.nan))
    cer_v = as_float(row.get("cer", np.nan))
    asr_norm = str(row.get("asr_text_norm_scored", row.get("asr_text_norm", "")) or "").strip()
    sim_v = as_float(row.get("wavlm_sim", np.nan))

    if not asr_norm:
        reasons.append("ASR empty or unintelligible")
    if np.isfinite(wer_v):
        if wer_v >= 1.0:
            reasons.append("no word overlap with target")
        elif wer_v >= 0.7:
            reasons.append("severe word mismatch")
    if np.isfinite(cer_v) and cer_v >= 0.5:
        reasons.append("high character mismatch")
    if np.isfinite(sim_v) and float(sim_v) < 0.35:
        reasons.append("low speaker similarity")

    if not reasons:
        if np.isfinite(wer_v):
            if wer_v >= 0.3:
                reasons.append("moderate transcription mismatch")
            else:
                reasons.append("minor mismatch")
        else:
            reasons.append("unknown")
    return "; ".join(reasons)


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=getattr(logging, args.log_level), force=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime_device = resolve_device(args.device)

    if args.prompt_mode == "fixed" and (args.prompt_wav is None or args.prompt_text is None):
        raise ValueError("When --prompt-mode fixed, both --prompt-wav and --prompt-text are required.")
    if args.prompt_mode == "fixed":
        prompt_path = args.prompt_wav.expanduser().resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"--prompt-wav not found: {prompt_path}")
        args.prompt_wav = prompt_path

    checkpoint_path, model_config_path, token_file = resolve_model_paths(args)
    loaded_samples = load_samples(args.dataset_tsv, args.lang, args.max_samples)
    if args.min_target_seconds <= 0:
        raise ValueError("--min-target-seconds must be > 0.")
    if args.max_target_seconds <= 0:
        raise ValueError("--max-target-seconds must be > 0.")
    if args.min_target_seconds > args.max_target_seconds:
        raise ValueError("--min-target-seconds cannot be greater than --max-target-seconds.")
    sample_count = args.number_sentences if args.number_sentences > 0 else 50
    sample_duration_by_idx: dict[int, float] = {s.idx: prompt_duration_seconds(s) for s in loaded_samples}
    prompt_samples, _, in_target_count = select_samples_by_duration(loaded_samples, sample_count)
    generation_max_duration = min(float(args.max_duration), float(args.max_target_seconds))
    if args.prompt_mode == "self":
        prompt_ids = {s.idx for s in prompt_samples}
        target_samples, target_eligible_count = select_other_targets(
            loaded_samples,
            prompt_ids,
            sample_count,
            args.seed,
            args.min_target_seconds,
            args.max_target_seconds,
            sample_duration_by_idx,
        )
        if not target_samples:
            target_samples, target_eligible_count = select_other_targets(
                loaded_samples,
                set(),
                sample_count,
                args.seed,
                args.min_target_seconds,
                args.max_target_seconds,
                sample_duration_by_idx,
            )
        if not target_samples:
            raise ValueError(
                f"No target rows satisfy --min-target-seconds={args.min_target_seconds} "
                f"and --max-target-seconds={args.max_target_seconds}. "
                "Increase the limit or disable strict selection."
            )
        pair_count = min(len(prompt_samples), len(target_samples))
        eval_pairs = list(zip(prompt_samples[:pair_count], target_samples[:pair_count]))
    else:
        target_samples, target_eligible_count = select_other_targets(
            loaded_samples,
            set(),
            sample_count,
            args.seed,
            args.min_target_seconds,
            args.max_target_seconds,
            sample_duration_by_idx,
        )
        if not target_samples:
            raise ValueError(
                f"No target rows satisfy --min-target-seconds={args.min_target_seconds} "
                f"and --max-target-seconds={args.max_target_seconds}."
            )
        prompt_samples = target_samples
        eval_pairs = list(zip(prompt_samples, target_samples))

    args.generated_dir.mkdir(parents=True, exist_ok=True)
    prompt_cache_dir = args.generated_dir / "_prompt_cache"
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_tts_bundle(args, checkpoint_path, model_config_path, token_file, runtime_device)
    asr = make_asr(args, runtime_device)
    sim_scorer = make_similarity_scorer(args, runtime_device)

    tts_param_device = next(bundle["model"].parameters()).device
    vocoder_param_device = next(bundle["vocoder"].parameters()).device
    asr_param_device = next(asr.model.parameters()).device

    if runtime_device.type == "cuda":
        if tts_param_device.type != "cuda" or vocoder_param_device.type != "cuda" or asr_param_device.type != "cuda":
            raise RuntimeError(
                f"CUDA requested but models are not fully on CUDA: "
                f"TTS={tts_param_device}, vocoder={vocoder_param_device}, ASR={asr_param_device}"
            )

    print("=== Evaluation Config ===")
    print(f"Dataset TSV     : {args.dataset_tsv}")
    print(f"Language        : {normalize_lang(args.lang)}")
    print(f"Loaded samples  : {len(loaded_samples)}")
    print(f"Prompt samples  : {len(prompt_samples)}")
    print(f"Target samples  : {len(target_samples)}")
    print(f"Samples         : {len(eval_pairs)}")
    print(f"Target duration : 1.0-3.0s ({in_target_count}/{len(loaded_samples)} already in range)")
    print(f"Target secs     : {args.min_target_seconds:.2f}-{args.max_target_seconds:.2f}s ({target_eligible_count} eligible)")
    print(f"Gen max secs    : {generation_max_duration:.2f}s")
    print(f"Selection       : prompt via duration, target from separate pool (--number-sentences={sample_count})")
    print(f"Prompt mode     : {args.prompt_mode}")
    print(f"Model type      : {bundle['model_type']}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Model config    : {model_config_path}")
    print(f"Token file      : {token_file}")
    print(f"Tokenizer       : {args.tokenizer} (lang={bundle['espeak_lang']})")
    print(f"Device          : {bundle['device']}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"CUDA visible    : {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"CUDA count      : {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"TTS device      : {tts_param_device}")
    print(f"Vocoder device  : {vocoder_param_device}")
    print(f"ASR device      : {asr_param_device}")
    print(f"ASR language    : {args.asr_language} (auto: sr->hr, ar->ar)")
    if sim_scorer is None:
        print("SIM metric      : skipped")
    else:
        print(f"SIM metric      : {args.sim_model_name} ({args.sim_checkpoint})")
        if getattr(args, "_sim_upstream_ckpt_used", None):
            print(f"SIM upstream    : {args._sim_upstream_ckpt_used}")
        print(f"SIM device      : {'cuda:' + str(sim_scorer.device) if sim_scorer.use_gpu else 'cpu'}")
    print(f"Generated wavs  : {args.generated_dir}")
    print("")

    rows: list[dict[str, Any]] = []
    for prompt_sample, target_sample in tqdm(eval_pairs, desc="Evaluating", unit="sample"):
        sample_id = safe_id(target_sample.utt_id, f"sample_{target_sample.idx:06d}")
        gen_wav = args.generated_dir / f"{sample_id}.wav"
        ref_norm = norm_text(target_sample.text, target_sample.lang)

        row: dict[str, Any] = {
            "utt_id": target_sample.utt_id,
            "lang": target_sample.lang,
            "text": target_sample.text,
            "ref_text_norm": ref_norm,
            "prompt_utt_id": prompt_sample.utt_id,
            "prompt_text": "",
            "prompt_text_norm": "",
            "source_wav": str(prompt_sample.wav_path),
            "target_source_wav": str(target_sample.wav_path),
            "start": prompt_sample.start,
            "end": prompt_sample.end,
            "target_start": target_sample.start,
            "target_end": target_sample.end,
            "prompt_seconds": float(sample_duration_by_idx.get(prompt_sample.idx, np.nan)),
            "target_seconds": float(sample_duration_by_idx.get(target_sample.idx, np.nan)),
            "tts_wav": str(gen_wav),
            "status": "ok",
            "error": "",
            "sim_error": "",
        }
        try:
            if args.prompt_mode == "fixed":
                prompt_wav = args.prompt_wav.expanduser().resolve()
                prompt_text = args.prompt_text
                source_eval_wav = prompt_wav
            else:
                source_eval_wav = ensure_prompt_wav(prompt_sample, prompt_cache_dir)
                prompt_wav = source_eval_wav
                prompt_text = prompt_sample.text
            row["prompt_text"] = prompt_text
            row["prompt_text_norm"] = norm_text(prompt_text, target_sample.lang)
            row["source_eval_wav"] = str(source_eval_wav)

            with torch.inference_mode():
                gen_metrics = generate_sentence(
                    save_path=str(gen_wav),
                    prompt_text=prompt_text,
                    prompt_wav=str(prompt_wav),
                    text=target_sample.text,
                    model=bundle["model"],
                    vocoder=bundle["vocoder"],
                    tokenizer=bundle["tokenizer"],
                    feature_extractor=bundle["feature_extractor"],
                    device=bundle["device"],
                    num_step=args.num_step,
                    guidance_scale=args.guidance_scale,
                    speed=args.speed,
                    t_shift=args.t_shift,
                    target_rms=args.target_rms,
                    feat_scale=args.feat_scale,
                    sampling_rate=bundle["sampling_rate"],
                    max_duration=generation_max_duration,
                    remove_long_sil=args.remove_long_sil,
                )

            asr_out = asr(
                str(gen_wav),
                generate_kwargs={
                    "language": whisper_lang(target_sample.lang, args.asr_language),
                    "task": "transcribe",
                    "max_new_tokens": args.max_new_tokens,
                    "num_beams": args.num_beams,
                },
            )
            hyp_text = str(asr_out.get("text", ""))
            hyp_norm = norm_text(hyp_text, target_sample.lang)
            hyp_norm_scored = collapse_repeated_patterns(hyp_norm)
            row["asr_text_raw"] = hyp_text
            row["asr_text_norm"] = hyp_norm
            row["asr_text_norm_scored"] = hyp_norm_scored
            row["wer"] = safe_wer(ref_norm, hyp_norm_scored)
            row["cer"] = safe_cer(ref_norm, hyp_norm_scored)
            row["rtf"] = float(gen_metrics["rtf"])
            row["wav_seconds"] = float(gen_metrics["wav_seconds"])
            row["wavlm_sim"] = None
        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)
            row["asr_text_raw"] = ""
            row["asr_text_norm"] = ""
            row["asr_text_norm_scored"] = ""
            row["wer"] = np.nan
            row["cer"] = np.nan
            row["rtf"] = np.nan
            row["wav_seconds"] = np.nan
            row["wavlm_sim"] = np.nan
        rows.append(row)

    if sim_scorer is not None:
        # Free ASR/TTS GPU memory before speaker similarity scoring.
        asr = None
        bundle["model"] = None
        bundle["vocoder"] = None
        bundle["feature_extractor"] = None
        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()

        for row in tqdm(rows, desc="Scoring SIM", unit="sample"):
            if row.get("status") != "ok":
                row["wavlm_sim"] = np.nan
                continue
            try:
                row["wavlm_sim"] = sim_scorer.score(Path(row["source_eval_wav"]), Path(row["tts_wav"]))
            except Exception as e:
                row["wavlm_sim"] = np.nan
                row["sim_error"] = str(e)

    outdf = pd.DataFrame(rows)

    # Keep CSV concise: one prompt reference audio column + target text + metrics.
    csv_df = outdf.copy()
    csv_df["reference_wav"] = csv_df["source_eval_wav"] if "source_eval_wav" in csv_df.columns else csv_df.get("source_wav")
    csv_df["reference_text"] = csv_df.get("prompt_text_norm")
    csv_df["target_text"] = csv_df.get("text")
    csv_df["target_text_norm"] = csv_df.get("ref_text_norm")
    csv_df["asr_output_norm"] = csv_df.get("asr_text_norm_scored")
    csv_df["output_wav"] = csv_df.get("tts_wav")
    csv_columns = [
        "utt_id",
        "prompt_utt_id",
        "lang",
        "reference_wav",
        "reference_text",
        "target_text",
        "target_text_norm",
        "output_wav",
        "asr_output_norm",
        "prompt_seconds",
        "target_seconds",
        "wer",
        "cer",
        "rtf",
        "wav_seconds",
        "wavlm_sim",
        "status",
        "error",
        "sim_error",
    ]
    csv_columns = [c for c in csv_columns if c in csv_df.columns]
    csv_df[csv_columns].to_csv(args.out_csv, index=False)

    ok_df = outdf[outdf["status"] == "ok"].copy()
    err_df = outdf[outdf["status"] != "ok"].copy()
    metrics = ["wer", "cer", "rtf", "wav_seconds"]
    if not args.skip_sim:
        metrics.append("wavlm_sim")

    print("=== Run Summary ===")
    print(f"Total samples    : {len(outdf)}")
    print(f"Succeeded        : {len(ok_df)}")
    print(f"Failed           : {len(err_df)}")

    if len(ok_df) > 0:
        print("\n=== Global Metrics (successful samples) ===")
        global_summary = ok_df[metrics].agg(["mean", "median", "std", "min", "max"])
        with pd.option_context("display.max_rows", 100, "display.max_columns", 200):
            print(global_summary.round(2))

        print("\n=== Per-language Metrics ===")
        lang_summary = ok_df.groupby("lang")[metrics].agg(["count", "mean", "median", "std"]).sort_index()
        with pd.option_context("display.max_rows", 100, "display.max_columns", 200):
            print(lang_summary.round(2))

        n_bad = min(args.show_worst, len(ok_df))
        if n_bad > 0:
            print(f"\n=== Worst {n_bad} samples by WER ===")
            worst_df = ok_df.sort_values("wer", ascending=False).head(n_bad).copy()
            worst_df["why_bad"] = worst_df.apply(worst_reason, axis=1)
            cols = [
                # "utt_id",
                # "lang",
                # "why_bad",
                # "text",
                "ref_text_norm",
                "asr_text_norm_scored",
                "wer",
                # "cer",
                # "tts_wav",
            ]
            if not args.skip_sim:
                cols[9:9] = ["wavlm_sim"]
            rename_map = {
                "text": "target_text",
                "ref_text_norm": "target_text_norm",
                "asr_text_norm_scored": "asr_output_norm",
            }
            display_df = worst_df[cols].rename(columns=rename_map).copy()
            for c in ["target_text_norm", "asr_output_norm", "why_bad", "tts_wav"]:
                if c in display_df.columns:
                    display_df[c] = display_df[c].map(lambda x: trunc_text(x, 72))
            for c in ["wer", "cer", "wavlm_sim", "rtf", "wav_seconds"]:
                if c in display_df.columns:
                    display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(2)
            with pd.option_context("display.max_columns", 200, "display.width", 220, "display.colheader_justify", "left"):
                print(display_df.to_string(index=False))
    else:
        print("\nNo successful samples, so metric summaries are empty.")
        global_summary = pd.DataFrame()
        lang_summary = pd.DataFrame()

    json_obj = {
        "config": {
            "dataset_tsv": str(args.dataset_tsv),
            "lang": normalize_lang(args.lang),
            "max_samples": args.max_samples,
            "number_sentences": args.number_sentences,
            "min_target_seconds": args.min_target_seconds,
            "max_target_seconds": args.max_target_seconds,
            "generation_max_duration": generation_max_duration,
            "prompt_mode": args.prompt_mode,
            "prompt_wav": str(args.prompt_wav) if args.prompt_wav else None,
            "checkpoint_path": str(checkpoint_path),
            "model_config_path": str(model_config_path),
            "token_file": str(token_file),
            "model_type": bundle["model_type"],
            "tokenizer": args.tokenizer,
            "espeak_lang": bundle["espeak_lang"],
            "device": str(bundle["device"]),
            "asr_model_id": args.asr_model_id,
            "asr_language": args.asr_language,
            "sim_model_name": args.sim_model_name,
            "sim_checkpoint": str(args.sim_checkpoint),
            "sim_wavlm_ckpt": str(args.sim_wavlm_ckpt) if args.sim_wavlm_ckpt else None,
            "sim_wavlm_ckpt_used": getattr(args, "_sim_upstream_ckpt_used", None),
            "skip_sim": args.skip_sim,
            "generated_dir": str(args.generated_dir),
        },
        "counts": {
            "total": int(len(outdf)),
            "ok": int(len(ok_df)),
            "error": int(len(err_df)),
        },
        "summary_global": {m: metric_summary(ok_df[m].dropna()) for m in metrics if m in ok_df.columns},
        "summary_by_lang": {},
        "errors": err_df[["utt_id", "error"]].to_dict(orient="records"),
        "results": [{k: to_plain(v) for k, v in record.items()} for record in outdf.to_dict(orient="records")],
    }

    if len(ok_df) > 0:
        for lang_key, sub in ok_df.groupby("lang"):
            json_obj["summary_by_lang"][lang_key] = {m: metric_summary(sub[m].dropna()) for m in metrics if m in sub.columns}

    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)

    print("\n=== Output Files ===")
    print(f"Detailed CSV     : {args.out_csv}")
    print(f"Detailed JSON    : {args.out_json}")
    print(f"Generated wavs   : {args.generated_dir}")


if __name__ == "__main__":
    main()
