#!/bin/bash

# Fine-tune ZipVoice base model for Serbian with pretrained warm-start,
# adapting checkpoint to the target tokenizer vocab size and filtering
# out bad fbank cuts (NaN/Inf or extreme token/frame ratios).

export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*"
}

stage=0
stop_stage=8

nj=${NJ:-24}
lang=sr
prefix=serbian

train_tsv=data/raw/${prefix}_train.tsv
dev_tsv=data/raw/${prefix}_dev.tsv

manifest_dir=data/manifests
fbank_dir=data/fbank

train_cuts_raw=${manifest_dir}/${prefix}_cuts_train.jsonl.gz
dev_cuts_raw=${manifest_dir}/${prefix}_cuts_dev.jsonl.gz
train_cuts_tok=${manifest_dir}/${prefix}_tok_cuts_train.jsonl.gz
dev_cuts_tok=${manifest_dir}/${prefix}_tok_cuts_dev.jsonl.gz

train_cuts_tok_filt=${manifest_dir}/${prefix}_tok_filtered_cuts_train.jsonl.gz
dev_cuts_tok_filt=${manifest_dir}/${prefix}_tok_filtered_cuts_dev.jsonl.gz

train_fbank=${fbank_dir}/${prefix}_tok_filtered_cuts_train.jsonl.gz
dev_fbank=${fbank_dir}/${prefix}_tok_filtered_cuts_dev.jsonl.gz

train_fbank_clean=${fbank_dir}/${prefix}_tok_filtered_cuts_train.clean.jsonl.gz
dev_fbank_clean=${fbank_dir}/${prefix}_tok_filtered_cuts_dev.clean.jsonl.gz

serbian_tokens=data/tokens_${prefix}_espeak_360.txt
target_vocab_size=360

max_len=30
max_duration=80
base_lr=0.0001
condition_drop_ratio=0.05
num_buckets=${NUM_BUCKETS:-60}
num_workers=${NUM_WORKERS:-8}
world_size=1
num_epochs=100

min_frames_per_token=1.0

pretrained_repo="k2-fsa/ZipVoice"
pretrained_subdir="zipvoice"
pretrained_dir="exp/pretrained_zipvoice"

use_pretrained="${USE_PRETRAINED:-1}"
finetune_mode="${FINETUNE_MODE:-1}"
disable_grad_mod="${DISABLE_GRAD_MOD:-1}"
simple_swoosh="${SIMPLE_SWOOSH:-1}"
inf_check="${INF_CHECK:-0}"
inf_detail="${INF_CHECK_DETAIL:-0}"
inf_limit="${INF_CHECK_LIMIT:-0}"
detect_anomaly="${DETECT_ANOMALY:-0}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
teacher_ckpt="${TEACHER_CKPT:-${pretrained_dir}/model.pt}"
teacher_config="${TEACHER_CONFIG:-${pretrained_dir}/model.json}"
adapted_ckpt="${pretrained_dir}/model_espeak_init.pt"

if [ "${use_pretrained}" -eq 0 ]; then
  teacher_ckpt=""
  teacher_config="egs/zipvoice/conf/zipvoice_base.json"
fi

log "Configured: prefix=${prefix}, lang=${lang}, num_epochs=${num_epochs}, base_lr=${base_lr}, use_pretrained=${use_pretrained}, finetune_mode=${finetune_mode}, disable_grad_mod=${disable_grad_mod}, simple_swoosh=${simple_swoosh}, inf_check=${inf_check}, inf_detail=${inf_detail}, inf_limit=${inf_limit}, detect_anomaly=${detect_anomaly}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ "${use_pretrained}" -eq 1 ]; then
  if [ ! -f "${teacher_ckpt}" ] || [ ! -f "${teacher_config}" ]; then
    log "Stage 0: downloading pretrained ZipVoice base to ${pretrained_dir}"
    mkdir -p "${pretrained_dir}"
    PRETRAINED_REPO="${pretrained_repo}" PRETRAINED_SUBDIR="${pretrained_subdir}" PRETRAINED_DIR="${pretrained_dir}" \
    python3 - <<'PY'
import os
import shutil
from huggingface_hub import hf_hub_download

repo = os.environ["PRETRAINED_REPO"]
subdir = os.environ["PRETRAINED_SUBDIR"]
out_dir = os.environ["PRETRAINED_DIR"]

for fname in ("model.pt", "model.json", "tokens.txt"):
    src = hf_hub_download(repo, filename=f"{subdir}/{fname}")
    dst = os.path.join(out_dir, fname)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
PY
  else
    log "Stage 0: pretrained files already present, skipping download"
  fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: creating manifests from TSV files"
  for subset in train dev; do
    if [ "${subset}" = "train" ]; then
      tsv_path="${train_tsv}"
    else
      tsv_path="${dev_tsv}"
    fi
    [ -f "$tsv_path" ] || { echo "Missing $tsv_path" >&2; exit 1; }
    python3 -m zipvoice.bin.prepare_dataset \
      --tsv-path ${tsv_path} \
      --prefix ${prefix} \
      --subset ${subset} \
      --num-jobs ${nj} \
      --output-dir ${manifest_dir}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: adding espeak tokens to manifests"
  for subset in train dev; do
    log "  tokens: subset=${subset}"
    if [ "${subset}" = "train" ]; then
      in_file=${train_cuts_raw}
      out_file=${train_cuts_tok}
    else
      in_file=${dev_cuts_raw}
      out_file=${dev_cuts_tok}
    fi
    python3 -m zipvoice.bin.prepare_tokens \
      --input-file ${in_file} \
      --output-file ${out_file} \
      --tokenizer espeak \
      --lang ${lang}
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Stage 2.5: filtering empty token cuts"
  python3 egs/zipvoice/local/filter_empty_tokens.py \
    --input "${train_cuts_tok}" \
    --output "${train_cuts_tok_filt}"
  python3 egs/zipvoice/local/filter_empty_tokens.py \
    --input "${dev_cuts_tok}" \
    --output "${dev_cuts_tok_filt}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Stage 3: computing fbank for dataset ${prefix}_tok_filtered"
  for subset in train dev; do
    python3 -m zipvoice.bin.compute_fbank \
      --source-dir ${manifest_dir} \
      --dest-dir ${fbank_dir} \
      --dataset ${prefix}_tok_filtered \
      --subset ${subset} \
      --num-jobs ${nj}
  done
fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#   log "Stage 3.5: filtering bad fbank cuts"
#   python3 egs/zipvoice/local/filter_bad_fbank.py \
#     --input "${train_fbank}" \
#     --output "${train_fbank_clean}" \
#     --min-frames-per-token ${min_frames_per_token}
#   python3 egs/zipvoice/local/filter_bad_fbank.py \
#     --input "${dev_fbank}" \
#     --output "${dev_fbank_clean}" \
#     --min-frames-per-token ${min_frames_per_token}
# fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  log "Stage 4: patching tokens.txt using pretrained vocab"
  python3 egs/zipvoice/local/patch_tokens_pretrained.py \
    --manifest ${train_cuts_tok_filt} \
    --pretrained-tokens ${pretrained_dir}/tokens.txt \
    --output ${serbian_tokens} \
    --strict 1 \
    --fill-missing 1 \
    --target-vocab-size ${target_vocab_size}
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && [ "${use_pretrained}" -eq 1 ]; then
  log "Stage 4.5: skipping vocab adaptation (using pretrained vocab size)"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export ZIPVOICE_DISABLE_GRAD_MOD="${disable_grad_mod}"
  export ZIPVOICE_SIMPLE_SWOOSH="${simple_swoosh}"
  export ZIPVOICE_INF_CHECK_DETAIL="${inf_detail}"
  export ZIPVOICE_INF_CHECK_LIMIT="${inf_limit}"
  export ZIPVOICE_DETECT_ANOMALY="${detect_anomaly}"
  extra_args=()
  if [ -n "${resume_checkpoint}" ]; then
    extra_args=(--checkpoint "${resume_checkpoint}" --model-config "${teacher_config}" --finetune ${finetune_mode} --base-lr 0.0001)
  elif [ -n "${teacher_ckpt}" ]; then
    extra_args=(--checkpoint "${teacher_ckpt}" --model-config "${teacher_config}" --finetune ${finetune_mode} --base-lr 0.0001)
  fi

  log "Stage 5: training ZipVoice base model"
  train_cmd=(
    python3 -m zipvoice.bin.train_zipvoice
    --world-size ${world_size}
    --use-fp16 0
    --num-iters 0
    --num-epochs ${num_epochs}
    --max-duration ${max_duration}
    --max-len ${max_len}
    --base-lr ${base_lr}
    --condition-drop-ratio ${condition_drop_ratio}
    --num-buckets ${num_buckets}
    --num-workers ${num_workers}
    --model-config ${teacher_config}
    --tokenizer espeak
    --lang ${lang}
    --token-file ${serbian_tokens}
    --dataset custom
    --train-manifest ${train_fbank}
    --dev-manifest ${dev_fbank}
    --exp-dir exp/zipvoice_${prefix}
    --inf-check ${inf_check}
  )
  if [ "${#extra_args[@]}" -ne 0 ]; then
    train_cmd+=("${extra_args[@]}")
  fi
  log "Training command: ${train_cmd[*]}"
  "${train_cmd[@]}"
fi
