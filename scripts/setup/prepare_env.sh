#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/k2-fsa/ZipVoice.git"
REPO_DIR="ZipVoice-master"
APP_REQ_FILE="requirements.txt"
PKG_FILE="packages.txt"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"

echo "[1/6] Cloning/updating ZipVoice..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
elif [ -d "$REPO_DIR" ]; then
  echo "Directory '$REPO_DIR' exists but is not a git repo. Reusing it as-is."
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[2/6] Installing system dependencies from packages.txt..."
if [ -f "$PKG_FILE" ] && command -v apt-get >/dev/null 2>&1; then
  mapfile -t system_packages < <(sed 's/#.*$//' "$PKG_FILE" | awk 'NF')
  if [ "${#system_packages[@]}" -gt 0 ]; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y "${system_packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${system_packages[@]}"
    else
      echo "Warning: apt-get found but no root/sudo; skip system package install."
    fi
  fi
else
  echo "Skipping local system package install (Spaces uses packages.txt automatically)."
fi

echo "[3/6] Installing Python requirements..."
if [ ! -f "$APP_REQ_FILE" ]; then
  echo "Error: '$APP_REQ_FILE' not found."
  exit 1
fi
$PIP_BIN install --upgrade pip
$PIP_BIN install -r "$APP_REQ_FILE"

echo "[4/6] Using ZipVoice directly from repository (no pip install)..."
if [ ! -d "$REPO_DIR/zipvoice" ]; then
  echo "Error: '$REPO_DIR/zipvoice' not found."
  exit 1
fi
$PYTHON_BIN - <<'PY'
import sys
from pathlib import Path

repo = Path("ZipVoice-master").resolve()
sys.path.insert(0, str(repo))
import zipvoice
print(f"zipvoice import OK from repo: {zipvoice.__file__}")
PY

echo "[5/6] Downloading HF model folders into exp/..."
mkdir -p exp

$PYTHON_BIN - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

targets = [
    ("karim1993/zipvoice-ar-finetuned", "zipvoice_ar"),
    ("karim1993/zipvoice-ar-finetuned", "zipvoice_distill_ar"),
    ("karim1993/zipvoice-sr-finetuned", "zipvoice_distill_sr"),
    ("karim1993/zipvoice-sr-finetuned", "zipvoice_sr"),
]

for repo_id, folder in targets:
    print(f"Downloading {repo_id}/{folder} -> exp/{folder}")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{folder}/*"],
        local_dir="exp",
        local_dir_use_symlinks=False,
        resume_download=True,
    )

# Prefetch vocoder to default HF cache (~/.cache/huggingface/hub by default).
# app.py -> get_vocoder(None) -> Vocos.from_pretrained("charactr/vocos-mel-24khz")
print("Prefetching vocoder repo charactr/vocos-mel-24khz into HF cache")
hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="config.yaml")
hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="pytorch_model.bin")
PY

echo "[6/6] Environment is ready."
