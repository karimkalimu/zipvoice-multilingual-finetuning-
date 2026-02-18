# app.py

"""
Gradio inference UI with a fixed model catalog.

Run:
    CUDA_VISIBLE_DEVICES="" python -m app
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("ZIPVOICE_HF_HUB_OFFLINE", "0"))

import datetime as dt
import gc
import json
import logging
import shutil
import sys
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import gradio as gr
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _download_zipvoice_source(dst_dir: Path) -> None:
    if dst_dir.is_dir():
        return
    if not _is_truthy(os.getenv("ZIPVOICE_AUTO_DOWNLOAD_CODE", "1")):
        return

    code_url = os.getenv(
        "ZIPVOICE_CODE_URL",
        "https://github.com/k2-fsa/ZipVoice/archive/refs/heads/master.zip",
    )
    tmp_zip = dst_dir.parent / "_zipvoice_source.zip"
    tmp_extract_dir = dst_dir.parent / "_zipvoice_extract"

    try:
        logging.info("Downloading ZipVoice source from %s", code_url)
        with urllib.request.urlopen(code_url) as response, open(tmp_zip, "wb") as out_f:
            out_f.write(response.read())
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_extract_dir)

        extracted_candidates = sorted(tmp_extract_dir.glob("ZipVoice-*"))
        if not extracted_candidates:
            raise RuntimeError("Downloaded ZipVoice archive did not contain ZipVoice-* directory.")
        extracted_dir = extracted_candidates[0]

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.move(str(extracted_dir), str(dst_dir))
        logging.info("ZipVoice source prepared at %s", dst_dir)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()
        if tmp_extract_dir.exists():
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)


if "zipvoice" not in sys.modules:
    _zipvoice_dir_env = os.getenv("ZIPVOICE_CODE_DIR", "").strip()
    _zipvoice_master = Path(_zipvoice_dir_env) if _zipvoice_dir_env else Path(__file__).resolve().parent / "ZipVoice-master"
    if not _zipvoice_master.is_dir():
        _download_zipvoice_source(_zipvoice_master)
    if not _zipvoice_master.is_dir():
        raise RuntimeError(
            f"ZipVoice source is missing at '{_zipvoice_master}'. "
            "Set ZIPVOICE_CODE_DIR or enable ZIPVOICE_AUTO_DOWNLOAD_CODE=1."
        )
    _zipvoice_master_str = str(_zipvoice_master)
    if _zipvoice_master_str not in sys.path:
        sys.path.insert(0, _zipvoice_master_str)


from zipvoice.bin.infer_zipvoice import generate_sentence as generate_sentence_torch, get_vocoder
from zipvoice.bin.infer_zipvoice_onnx import OnnxModel as BaseOnnxModel
from zipvoice.bin.infer_zipvoice_onnx import sample as onnx_sample
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer, EspeakTokenizer, LibriTTSTokenizer, SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import add_punctuation, chunk_tokens_punctuation, cross_fade_concat, load_prompt_wav, remove_silence, rms_norm

# ---------------------------------------------------------------------------
# Configure your models here.
# ---------------------------------------------------------------------------
# Fields:
# - id: unique identifier used by the UI
# - name: display name
# - backend: "torch" or "onnx"
# - model_path: folder path
# - distill: True/False
# - description: text shown in the UI
# - tokenizer: emilia | espeak | simple | libritts
# - language: auto-filled espeak language (e.g. sr, ar)
# Optional fields:
# - checkpoint_name (torch): defaults to model.pt
# - onnx_int8 (onnx): defaults to False
# - vocoder_path: local vocoder directory (if omitted, uses HF vocoder)
MODEL_CATALOG: List[Dict[str, object]] = [
    {
        "id": "arabic_distill_torch",
        "name": "Arabic Distilled (Torch)",
        "backend": "torch",
        "model_path": "exp/zipvoice_distill_ar",
        "checkpoint_name": "checkpoint-2000.pt",
        "distill": True,
        "description": "Arabic distilled checkpoint (Torch).",
        "tokenizer": "espeak",
        "language": "ar",
        "recommended_min_num_step": 4,
        "default_num_step": 4,
        "default_guidance_scale": 3.0,
    },
    {
        "id": "arabic_distill_onnx",
        "name": "Arabic Distilled (ONNX)",
        "backend": "onnx",
        "model_path": "exp/zipvoice_distill_ar",
        "distill": True,
        "description": "Arabic distilled ONNX (text_encoder/fm_decoder).",
        "tokenizer": "espeak",
        "language": "ar",
        "recommended_min_num_step": 4,
        "default_num_step": 4,
        "default_guidance_scale": 3.0,
        "onnx_int8": True,
    },
    {
        "id": "serbian_distill_torch",
        "name": "Serbian Distilled (Torch)",
        "backend": "torch",
        "model_path": "exp/zipvoice_distill_sr",
        "checkpoint_name": "checkpoint-2000.pt",
        "distill": True,
        "description": "Serbian distilled checkpoint (Torch).",
        "tokenizer": "espeak",
        "language": "sr",
        "recommended_min_num_step": 4,
        "default_num_step": 4,
        "default_guidance_scale": 3.0,
    },
    {
        "id": "serbian_distill_onnx",
        "name": "Serbian Distilled (ONNX)",
        "backend": "onnx",
        "model_path": "exp/zipvoice_distill_sr",
        "distill": True,
        "description": "Serbian distilled ONNX (text_encoder/fm_decoder).",
        "tokenizer": "espeak",
        "language": "sr",
        "recommended_min_num_step": 4,
        "default_num_step": 4,
        "default_guidance_scale": 3.0,
        "onnx_int8": True,
    },
    {
        "id": "arabic_base_torch_best",
        "name": "Arabic Best Non-Distilled (Torch)",
        "backend": "torch",
        "model_path": "exp/zipvoice_ar",
        "checkpoint_name": "epoch-26-avg-4.pt",
        "distill": False,
        "description": "Arabic best non-distilled checkpoint.",
        "tokenizer": "espeak",
        "language": "ar",
        "recommended_min_num_step": 8,
        "default_num_step": 8,
        "default_guidance_scale": 1.0,
    },
    {
        "id": "serbian_base_torch_best",
        "name": "Serbian Best Non-Distilled (Torch)",
        "backend": "torch",
        "model_path": "exp/zipvoice_sr",
        "checkpoint_name": "epoch-15-avg-4.pt",
        "distill": False,
        "description": "Serbian best non-distilled checkpoint.",
        "tokenizer": "espeak",
        "language": "sr",
        "recommended_min_num_step": 8,
        "default_num_step": 8,
        "default_guidance_scale": 1.0,
    },
]

MAX_EXAMPLES = 3
DEFAULTS = {
    "num_step": 16,
    "guidance_scale": 1.0,
    "speed": 1.0,
    "t_shift": 0.5,
    "target_rms": 0.1,
    "feat_scale": 0.1,
    "max_duration": 15.0,
    "remove_long_sil": False,
}

LANGUAGE_EXAMPLES: Dict[str, Dict[str, str]] = {
    "ar": {
        "prompt_wav": "assets/arabic_prompt.wav",
        "prompt_text": "النَّصُّ المُطابِقُ لِلصَّوْتِ",
        "target_text": "النَّصُّ المُطابِقُ لِلصَّوْتِ",
    },
    "sr": {
        "prompt_wav": "assets/serbian_prompt.wav",
        "prompt_text": "a da se posle napravi sistem?",
        "target_text": "a da se posle napravi sistem?",
    },
}

LOG_LEVEL = os.getenv("ZIPVOICE_GRADIO_LOG_LEVEL", "INFO").upper()
RUNNING_ON_SPACE = bool(os.getenv("SPACE_ID"))
VOCODER_REPO_ID = "charactr/vocos-mel-24khz"
MODEL_SNAPSHOT_REPOS = {
    "zipvoice_ar": "karim1993/zipvoice-ar-finetuned",
    "zipvoice_distill_ar": "karim1993/zipvoice-ar-finetuned",
    "zipvoice_sr": "karim1993/zipvoice-sr-finetuned",
    "zipvoice_distill_sr": "karim1993/zipvoice-sr-finetuned",
}
_MODEL_BY_ID = {str(m["id"]): m for m in MODEL_CATALOG}
_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE = {"key": None, "bundle": None}


def _model_dir_ready(model_dir: Path) -> bool:
    return model_dir.is_dir() and (model_dir / "model.json").exists() and (model_dir / "tokens.txt").exists()


def _model_download_token() -> str | None:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    token = os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    return token or None


def _ensure_model_artifacts() -> None:
    missing = [Path(str(spec["model_path"])) for spec in MODEL_CATALOG if not _model_dir_ready(Path(str(spec["model_path"])))]
    if not missing:
        return

    auto_download = _is_truthy(os.getenv("ZIPVOICE_AUTO_DOWNLOAD_MODELS", "1" if RUNNING_ON_SPACE else "0"))
    if not auto_download:
        missing_str = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Missing model folders: {missing_str}. "
            "Enable ZIPVOICE_AUTO_DOWNLOAD_MODELS=1 or upload these folders."
        )

    token = _model_download_token()
    exp_root = Path("exp")
    exp_root.mkdir(parents=True, exist_ok=True)
    downloaded = set()
    for model_dir in missing:
        folder = model_dir.name
        repo_id = MODEL_SNAPSHOT_REPOS.get(folder)
        if not repo_id:
            raise RuntimeError(
                f"Missing model folder '{model_dir}' and no snapshot source is configured for '{folder}'."
            )
        key = (repo_id, folder)
        if key in downloaded:
            continue
        logging.info("Downloading '%s' from '%s' into '%s'...", folder, repo_id, exp_root)
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{folder}/*"],
            local_dir=str(exp_root),
            token=token,
            resume_download=True,
        )
        downloaded.add(key)

    unresolved = [Path(str(spec["model_path"])) for spec in MODEL_CATALOG if not _model_dir_ready(Path(str(spec["model_path"])))]
    if unresolved:
        unresolved_str = ", ".join(str(p) for p in unresolved)
        raise RuntimeError(
            f"Model download completed but required folders are still missing: {unresolved_str}"
        )


def _ensure_vocoder_available() -> None:
    if all(spec.get("vocoder_path") for spec in MODEL_CATALOG):
        return

    if _is_truthy(os.getenv("HF_HUB_OFFLINE", "0")):
        raise RuntimeError(
            "HF_HUB_OFFLINE=1 but MODEL_CATALOG has no local vocoder_path. "
            "Set ZIPVOICE_HF_HUB_OFFLINE=0 or configure vocoder_path for all models."
        )

    prefetch = _is_truthy(os.getenv("ZIPVOICE_PREFETCH_VOCODER", "1" if RUNNING_ON_SPACE else "0"))
    if not prefetch:
        return

    token = _model_download_token()
    for filename in ("config.yaml", "pytorch_model.bin"):
        hf_hub_download(repo_id=VOCODER_REPO_ID, filename=filename, token=token)


def _example_audio_exists(language: str) -> bool:
    prompt_wav = LANGUAGE_EXAMPLES.get(language, {}).get("prompt_wav", "")
    return bool(prompt_wav) and Path(prompt_wav).exists()


class FlexibleOnnxModel(BaseOnnxModel):
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        providers: List[str],
        num_thread: int = 1,
    ):
        self.providers = providers
        super().__init__(text_encoder_path=text_encoder_path, fm_decoder_path=fm_decoder_path, num_thread=num_thread)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=self.providers,
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=self.providers,
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=LOG_LEVEL,
        force=True,
    )


def _get_total_ram_gb() -> float:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                return (pages * page_size) / (1024**3)
    except Exception:
        pass
    return 0.0


def _decide_only_onnx_models(cuda_available: bool) -> bool:
    mode = os.getenv("ZIPVOICE_ONLY_ONNX_MODELS", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False

    if cuda_available:
        return False
    if RUNNING_ON_SPACE:
        return True

    threshold_gb = float(os.getenv("ZIPVOICE_ONLY_ONNX_IF_RAM_LESS_THAN_GB", "8"))
    total_ram_gb = _get_total_ram_gb()
    if total_ram_gb <= 0:
        logging.warning("Could not detect total RAM. Falling back to ONNX-only catalog on CPU hosts.")
        return True

    only_onnx = total_ram_gb < threshold_gb
    logging.info(
        "RAM check: total=%.2fGB threshold=%.2fGB -> only_onnx_models=%s",
        total_ram_gb,
        threshold_gb,
        only_onnx,
    )
    return only_onnx


def _model_defaults(spec: Dict[str, object]) -> Tuple[int, float]:
    is_distill = bool(spec.get("distill", False))
    default_num_step = int(spec.get("default_num_step", 4 if is_distill else 8))
    default_guidance = float(spec.get("default_guidance_scale", 3.0 if is_distill else 1.0))
    return default_num_step, default_guidance


def _model_choices(include_onnx: bool = True, only_onnx: bool = False) -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = []
    for model_id, spec in _MODEL_BY_ID.items():
        backend = str(spec.get("backend", "")).lower()
        if only_onnx and backend != "onnx":
            continue
        if not include_onnx and backend == "onnx":
            continue
        path = Path(str(spec["model_path"]))
        status = "OK" if path.exists() else "MISSING"
        label = f'{spec["name"]} | {spec["backend"]} | {status}'
        choices.append((label, model_id))
    return choices


def _example_visibility_for_language(language: str) -> Tuple[dict, dict]:
    lang_key = language.strip().lower()
    return (
        gr.update(visible=(lang_key == "ar" and _example_audio_exists("ar"))),
        gr.update(visible=(lang_key == "sr" and _example_audio_exists("sr"))),
    )


def _default_model_id(include_onnx: bool = True, only_onnx: bool = False) -> str:
    for model_id, spec in _MODEL_BY_ID.items():
        backend = str(spec.get("backend", "")).lower()
        if only_onnx and backend != "onnx":
            continue
        if not include_onnx and backend == "onnx":
            continue
        if Path(str(spec["model_path"])).exists():
            return model_id
    for model_id, spec in _MODEL_BY_ID.items():
        backend = str(spec.get("backend", "")).lower()
        if only_onnx and backend != "onnx":
            continue
        if include_onnx or backend != "onnx":
            return model_id
    raise ValueError("No models available for current backend filter.")


def _clear_cache() -> None:
    with _CACHE_LOCK:
        _RUNTIME_CACHE["key"] = None
        _RUNTIME_CACHE["bundle"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _create_tokenizer(name: str, token_file: Path, lang: str):
    if name == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    if name == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    if name == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=lang)
    if name == "simple":
        return SimpleTokenizer(token_file=token_file)
    raise ValueError(f"Unsupported tokenizer: {name}")


def _resolve_torch_device(device_choice: str) -> torch.device:
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise gr.Error("CUDA selected but no CUDA device is available.")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _resolve_onnx_providers(device_choice: str) -> Tuple[List[str], str]:
    if device_choice == "cuda":
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
        logging.warning("CUDA selected for ONNX, but CUDAExecutionProvider is unavailable. Falling back to CPU.")
    return ["CPUExecutionProvider"], "cpu"


def _load_torch_bundle(spec: Dict[str, object], lang: str, device_choice: str) -> Dict[str, object]:
    model_dir = Path(str(spec["model_path"]))
    checkpoint_name = str(spec.get("checkpoint_name", "model.pt"))
    checkpoint_path = model_dir / checkpoint_name
    model_config_path = model_dir / "model.json"
    token_file = model_dir / "tokens.txt"

    for p in [model_dir, model_config_path, token_file, checkpoint_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    tokenizer_name = str(spec.get("tokenizer", "espeak"))
    tokenizer = _create_tokenizer(tokenizer_name, token_file, lang)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    if bool(spec.get("distill", False)):
        model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)
    else:
        model = ZipVoice(**model_config["model"], **tokenizer_config)

    checkpoint_path_str = str(checkpoint_path)
    if checkpoint_path_str.endswith(".pt"):
        load_checkpoint(filename=checkpoint_path, model=model, strict=True)
    elif checkpoint_path_str.endswith(".safetensors"):
        import safetensors.torch

        safetensors.torch.load_model(model, checkpoint_path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    torch_device = _resolve_torch_device(device_choice)
    model = model.to(torch_device).eval()

    vocoder = get_vocoder(spec.get("vocoder_path"))
    vocoder = vocoder.to(torch_device).eval()

    if model_config["feature"]["type"] != "vocos":
        raise NotImplementedError(f'Unsupported feature type: {model_config["feature"]["type"]}')

    return {
        "backend": "torch",
        "device": torch_device,
        "model": model,
        "vocoder": vocoder,
        "tokenizer": tokenizer,
        "feature_extractor": VocosFbank(),
        "sampling_rate": int(model_config["feature"]["sampling_rate"]),
    }


def _onnx_model_files(spec: Dict[str, object], onnx_int8: bool) -> Tuple[Path, Path]:
    model_dir = Path(str(spec["model_path"]))
    text_encoder_name = "text_encoder_int8.onnx" if onnx_int8 else "text_encoder.onnx"
    fm_decoder_name = "fm_decoder_int8.onnx" if onnx_int8 else "fm_decoder.onnx"
    return model_dir / text_encoder_name, model_dir / fm_decoder_name


def _load_onnx_bundle(
    spec: Dict[str, object],
    lang: str,
    device_choice: str,
    onnx_int8: bool,
    num_thread: int,
) -> Dict[str, object]:
    model_dir = Path(str(spec["model_path"]))
    model_config_path = model_dir / "model.json"
    token_file = model_dir / "tokens.txt"
    text_encoder_path, fm_decoder_path = _onnx_model_files(spec, onnx_int8)

    for p in [model_dir, model_config_path, token_file, text_encoder_path, fm_decoder_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    if model_config["feature"]["type"] != "vocos":
        raise NotImplementedError(f'Unsupported feature type: {model_config["feature"]["type"]}')

    providers, onnx_runtime_device = _resolve_onnx_providers(device_choice)
    onnx_model = FlexibleOnnxModel(
        text_encoder_path=str(text_encoder_path),
        fm_decoder_path=str(fm_decoder_path),
        providers=providers,
        num_thread=num_thread,
    )

    tokenizer_name = str(spec.get("tokenizer", "espeak"))
    tokenizer = _create_tokenizer(tokenizer_name, token_file, lang)

    vocoder_device = _resolve_torch_device(onnx_runtime_device) if onnx_runtime_device == "cuda" else torch.device("cpu")
    vocoder = get_vocoder(spec.get("vocoder_path"))
    vocoder = vocoder.to(vocoder_device).eval()

    return {
        "backend": "onnx",
        "onnx_providers": providers,
        "onnx_runtime_device": onnx_runtime_device,
        "vocoder_device": vocoder_device,
        "model": onnx_model,
        "vocoder": vocoder,
        "tokenizer": tokenizer,
        "feature_extractor": VocosFbank(),
        "sampling_rate": int(model_config["feature"]["sampling_rate"]),
    }


def _get_bundle(
    model_id: str,
    lang: str,
    device_choice: str,
    onnx_int8: bool,
    num_thread: int,
) -> Dict[str, object]:
    spec = _MODEL_BY_ID[model_id]
    backend = str(spec["backend"])
    key = (model_id, backend, lang, device_choice, bool(onnx_int8), int(num_thread))

    with _CACHE_LOCK:
        if _RUNTIME_CACHE["key"] == key and _RUNTIME_CACHE["bundle"] is not None:
            logging.info("Using cached bundle for key=%s", key)
            return _RUNTIME_CACHE["bundle"]

        _RUNTIME_CACHE["key"] = None
        _RUNTIME_CACHE["bundle"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if backend == "onnx":
            bundle = _load_onnx_bundle(spec, lang=lang, device_choice=device_choice, onnx_int8=onnx_int8, num_thread=num_thread)
        elif backend == "torch":
            bundle = _load_torch_bundle(spec, lang=lang, device_choice=device_choice)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        _RUNTIME_CACHE["key"] = key
        _RUNTIME_CACHE["bundle"] = bundle
        return bundle


def _generate_sentence_onnx(
    save_path: str,
    prompt_text: str,
    prompt_wav_path: str,
    target_text: str,
    bundle: Dict[str, object],
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    target_rms: float,
    feat_scale: float,
    remove_long_sil: bool,
) -> Dict[str, float]:
    sampling_rate = int(bundle["sampling_rate"])
    model = bundle["model"]
    vocoder = bundle["vocoder"]
    tokenizer = bundle["tokenizer"]
    feature_extractor = bundle["feature_extractor"]
    vocoder_device = bundle["vocoder_device"]

    prompt_wav = load_prompt_wav(prompt_wav_path, sampling_rate=sampling_rate)
    prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    target_text = add_punctuation(target_text)
    prompt_text = add_punctuation(prompt_text)

    tokens_str = tokenizer.texts_to_tokens([target_text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]
    if len(prompt_tokens_str) == 0:
        raise ValueError("Reference text produced no prompt tokens.")

    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (len(prompt_tokens_str) * speed)
    max_tokens = max(1, int((25 - prompt_duration) / max(token_duration, 1e-6)))
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
    if not chunked_tokens_str:
        raise ValueError("Target text produced no synthesis chunks.")

    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    chunked_features = []
    start_t = dt.datetime.now()
    for tokens in chunked_tokens:
        pred_features = onnx_sample(
            model=model,
            tokens=[tokens],
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )
        pred_features = pred_features.permute(0, 2, 1) / feat_scale
        chunked_features.append(pred_features)

    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()
    for pred_features in chunked_features:
        pred_features = pred_features.to(vocoder_device)
        wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1).detach().cpu()
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        chunked_wavs.append(wav)

    final_wav = cross_fade_concat(chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate)
    final_wav = remove_silence(final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0)
    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)

    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    return {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": t / max(wav_seconds, 1e-6),
    }


def _run_infer(
    model_id: str,
    device_choice: str,
    prompt_wav: str,
    prompt_text: str,
    target_text: str,
    lang: str,
    onnx_int8: bool,
    onnx_num_thread: int,
    num_examples: int,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    target_rms: float,
    feat_scale: float,
    max_duration: float,
    remove_long_sil: bool,
):
    if not model_id:
        raise gr.Error("Please select a model.")
    if not prompt_wav:
        raise gr.Error("Please upload reference audio.")
    if not prompt_text:
        raise gr.Error("Please provide reference text.")
    if not target_text:
        raise gr.Error("Please provide target text.")

    model_spec = _MODEL_BY_ID[model_id]
    min_num_step = int(model_spec.get("recommended_min_num_step", 1))
    effective_num_step = max(int(num_step), min_num_step)
    bundle = _get_bundle(
        model_id=model_id,
        lang=lang,
        device_choice=device_choice,
        onnx_int8=onnx_int8,
        num_thread=onnx_num_thread,
    )
    backend = str(model_spec["backend"])
    num_examples = max(1, min(MAX_EXAMPLES, int(num_examples)))

    output_paths: List[str] = []
    metric_lines: List[str] = []
    current_tmp: str = ""
    try:
        for i in range(num_examples):
            fd, out_path = tempfile.mkstemp(prefix=f"zipvoice_{backend}_{i}_", suffix=".wav")
            os.close(fd)
            current_tmp = out_path

            if backend == "torch":
                metrics = generate_sentence_torch(
                    save_path=out_path,
                    prompt_text=prompt_text,
                    prompt_wav=prompt_wav,
                    text=target_text,
                    model=bundle["model"],
                    vocoder=bundle["vocoder"],
                    tokenizer=bundle["tokenizer"],
                    feature_extractor=bundle["feature_extractor"],
                    device=bundle["device"],
                    num_step=effective_num_step,
                    guidance_scale=float(guidance_scale),
                    speed=float(speed),
                    t_shift=float(t_shift),
                    target_rms=float(target_rms),
                    feat_scale=float(feat_scale),
                    sampling_rate=int(bundle["sampling_rate"]),
                    max_duration=float(max_duration),
                    remove_long_sil=bool(remove_long_sil),
                )
            else:
                metrics = _generate_sentence_onnx(
                    save_path=out_path,
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav,
                    target_text=target_text,
                    bundle=bundle,
                    num_step=effective_num_step,
                    guidance_scale=float(guidance_scale),
                    speed=float(speed),
                    t_shift=float(t_shift),
                    target_rms=float(target_rms),
                    feat_scale=float(feat_scale),
                    remove_long_sil=bool(remove_long_sil),
                )

            current_tmp = ""
            output_paths.append(out_path)
            metric_lines.append(
                f"example {i + 1}: rtf={metrics['rtf']:.4f}, duration={metrics['wav_seconds']:.2f}s"
            )
    except Exception as ex:
        if current_tmp and os.path.exists(current_tmp):
            os.unlink(current_tmp)
        for p in output_paths:
            if os.path.exists(p):
                os.unlink(p)
        raise gr.Error(f"Inference failed: {ex}")

    while len(output_paths) < MAX_EXAMPLES:
        output_paths.append(None)

    status = "\n".join(
        [
            f"Model: {model_spec['name']} ({backend})",
            f"Device selection: {device_choice}",
            f"Language: {lang}",
            f"num_step: requested={int(num_step)}, used={effective_num_step}",
            *metric_lines,
        ]
    )
    return output_paths[0], output_paths[1], output_paths[2], status


def _cpu_torch_note_update(model_id: str, device_choice: str):
    spec = _MODEL_BY_ID[model_id]
    backend = str(spec.get("backend", "")).lower()
    use_cpu = str(device_choice).strip().lower() == "cpu"
    return gr.update(visible=(backend == "torch" and use_cpu))


def _on_model_change(model_id: str, device_choice: str):
    spec = _MODEL_BY_ID[model_id]
    num_step, guidance = _model_defaults(spec)
    min_num_step = int(spec.get("recommended_min_num_step", 1))
    backend = str(spec["backend"])
    lang = str(spec.get("language", "en-us"))
    tokenizer = str(spec.get("tokenizer", "espeak"))
    description = str(spec.get("description", ""))
    ar_example_update, sr_example_update = _example_visibility_for_language(lang)
    ar_quality_note_update = gr.update(visible=(lang.strip().lower() == "ar"))
    return (
        description,
        str(min_num_step),
        backend,
        str(bool(spec.get("distill", False))),
        tokenizer,
        gr.update(value=lang, interactive=False),
        gr.update(value=num_step, minimum=min_num_step),
        guidance,
        ar_quality_note_update,
        ar_example_update,
        sr_example_update,
        gr.update(value=bool(spec.get("onnx_int8", False)), visible=(backend == "onnx")),
        gr.update(visible=(backend == "onnx")),
        _cpu_torch_note_update(model_id, device_choice),
    )


def build_app():
    _ensure_model_artifacts()
    _ensure_vocoder_available()

    cuda_available = torch.cuda.is_available()
    # On CPU-only hosts, enable ONNX-only catalog when RAM is below threshold.
    # Override with ZIPVOICE_ONLY_ONNX_MODELS=true/false.
    include_onnx_models = not cuda_available
    only_onnx_models = _decide_only_onnx_models(cuda_available=cuda_available)
    model_choices = _model_choices(include_onnx=include_onnx_models, only_onnx=only_onnx_models)
    available_model_ids = {model_id for _, model_id in model_choices}
    preferred_default_model_id = os.getenv("ZIPVOICE_DEFAULT_MODEL_ID", "serbian_distill_onnx").strip()
    default_model_id = (
        preferred_default_model_id
        if preferred_default_model_id in available_model_ids
        else _default_model_id(include_onnx=include_onnx_models, only_onnx=only_onnx_models)
    )
    default_spec = _MODEL_BY_ID[default_model_id]
    default_lang = str(default_spec.get("language", "en-us"))
    default_description = str(default_spec.get("description", ""))
    default_backend = str(default_spec["backend"])
    default_distill = str(bool(default_spec.get("distill", False)))
    default_tokenizer = str(default_spec.get("tokenizer", "espeak"))
    default_num_step, default_guidance = _model_defaults(default_spec)
    default_min_num_step = int(default_spec.get("recommended_min_num_step", 1))
    device_choices = ["cpu", "cuda"] if cuda_available else ["cpu"]
    default_device = "cuda" if cuda_available else "cpu"
    default_onnx_visible = default_backend == "onnx"
    default_cpu_torch_note_visible = default_backend.lower() == "torch" and default_device == "cpu"

    with gr.Blocks() as demo:
        gr.Markdown("# ZipVoice Gradio Infer ")

        with gr.Row():
            model_id = gr.Dropdown(
                choices=model_choices,
                value=default_model_id,
                label="Model",
            )
            device_choice = gr.Radio(
                choices=device_choices,
                value=default_device,
                label="Device",
            )
        cpu_torch_note = gr.Markdown(
            "Note: Torch models are slow on CPU. For faster CPU inference, use an ONNX model with `onnx_int8` enabled.",
            visible=default_cpu_torch_note_visible,
        )

        with gr.Accordion("Model Info", open=False):
            with gr.Row():
                backend = gr.Textbox(label="Backend", interactive=False, value=default_backend)
                distill = gr.Textbox(label="Distill", interactive=False, value=default_distill)
                tokenizer = gr.Textbox(label="Tokenizer", interactive=False, value=default_tokenizer)
                lang = gr.Textbox(label="Language (from model)", value=default_lang, interactive=False)
            with gr.Row():
                description = gr.Textbox(label="Description", interactive=False, value=default_description)
            with gr.Row():
                recommended_min_num_step = gr.Textbox(
                    label="Recommended min num_step",
                    interactive=False,
                    value=str(default_min_num_step),
                )

        with gr.Row():
            prompt_wav = gr.Audio(label="Reference Audio", type="filepath")
            prompt_text = gr.Textbox(label="Reference Text", lines=3, value="")

        target_text = gr.Textbox(
            label="Target Text",
            lines=4,
            value="",
        )
        default_lang_key = default_lang.strip().lower()
        ar_quality_note = gr.Markdown(
            "For better Arabic TTS quality, use diacritics (tashkeel) when possible.\n\n"
            "لجودة أفضل في تحويل النص العربي إلى كلام، يُفضَّل استخدام التشكيل قدر الإمكان.",
            visible=(default_lang_key == "ar"),
        )
        with gr.Group(visible=(default_lang_key == "ar" and _example_audio_exists("ar"))) as ar_examples_group:
            if _example_audio_exists("ar"):
                gr.Examples(
                    examples=[
                        [
                            LANGUAGE_EXAMPLES["ar"]["prompt_wav"],
                            LANGUAGE_EXAMPLES["ar"]["prompt_text"],
                            LANGUAGE_EXAMPLES["ar"]["target_text"],
                        ]
                    ],
                    inputs=[prompt_wav, prompt_text, target_text],
                    label="Arabic Example",
                )
        with gr.Group(visible=(default_lang_key == "sr" and _example_audio_exists("sr"))) as sr_examples_group:
            if _example_audio_exists("sr"):
                gr.Examples(
                    examples=[
                        [
                            LANGUAGE_EXAMPLES["sr"]["prompt_wav"],
                            LANGUAGE_EXAMPLES["sr"]["prompt_text"],
                            LANGUAGE_EXAMPLES["sr"]["target_text"],
                        ]
                    ],
                    inputs=[prompt_wav, prompt_text, target_text],
                    label="Serbian Example",
                )

        with gr.Row():
            num_examples = gr.Slider(1, MAX_EXAMPLES, value=1, step=1, label="Generated examples")
            num_step = gr.Slider(default_min_num_step, 16, value=default_num_step, step=1, label="num_step")
            guidance_scale = gr.Slider(0.5, 8.0, value=default_guidance, step=0.1, label="guidance_scale")
            speed = gr.Slider(0.5, 1.5, value=DEFAULTS["speed"], step=0.05, label="speed")

        with gr.Row():
            t_shift = gr.Slider(0.1, 1.0, value=DEFAULTS["t_shift"], step=0.05, label="t_shift")
            target_rms = gr.Slider(0.0, 0.3, value=DEFAULTS["target_rms"], step=0.01, label="target_rms")
            feat_scale = gr.Slider(0.01, 0.3, value=DEFAULTS["feat_scale"], step=0.01, label="feat_scale")
            max_duration = gr.Slider(5.0, 40.0, value=DEFAULTS["max_duration"], step=1.0, label="max_duration (torch)")

        with gr.Row():
            remove_long_sil = gr.Checkbox(value=DEFAULTS["remove_long_sil"], label="remove_long_sil")
            onnx_int8 = gr.Checkbox(
                value=bool(default_spec.get("onnx_int8", False)),
                label="onnx_int8",
                visible=default_onnx_visible,
            )
            onnx_num_thread = gr.Slider(1, 8, value=1, step=1, label="onnx_num_thread", visible=default_onnx_visible)

        generate_btn = gr.Button("Generate")

        with gr.Row():
            out_1 = gr.Audio(label="Generated 1", type="filepath")
            out_2 = gr.Audio(label="Generated 2", type="filepath")
            out_3 = gr.Audio(label="Generated 3", type="filepath")
        status = gr.Textbox(label="Run Summary", lines=6, interactive=False)

        model_id.change(
            _on_model_change,
            inputs=[model_id, device_choice],
            outputs=[
                description,
                recommended_min_num_step,
                backend,
                distill,
                tokenizer,
                lang,
                num_step,
                guidance_scale,
                ar_quality_note,
                ar_examples_group,
                sr_examples_group,
                onnx_int8,
                onnx_num_thread,
                cpu_torch_note,
            ],
        )
        device_choice.change(
            _cpu_torch_note_update,
            inputs=[model_id, device_choice],
            outputs=[cpu_torch_note],
        )
        generate_btn.click(
            _run_infer,
            inputs=[
                model_id,
                device_choice,
                prompt_wav,
                prompt_text,
                target_text,
                lang,
                onnx_int8,
                onnx_num_thread,
                num_examples,
                num_step,
                guidance_scale,
                speed,
                t_shift,
                target_rms,
                feat_scale,
                max_duration,
                remove_long_sil,
            ],
            outputs=[out_1, out_2, out_3, status],
        )

    return demo


if __name__ == "__main__":
    _setup_logging()
    app = build_app()
    app = app.queue(default_concurrency_limit=int(os.getenv("ZIPVOICE_CONCURRENCY", "1")))
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", os.getenv("ZIPVOICE_SERVER_PORT", "7860"))),
        show_error=True,
    )
