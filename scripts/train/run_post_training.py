# run_post_training.py
"""
Post-training runner for ZipVoice stages 3-8.

Stages:
  3: Average base ZipVoice checkpoints
  4: Distill (first stage)
  5: Average first-stage distill checkpoints
  6: Distill (second stage / re-distill)
  7: Export base ZipVoice ONNX
  8: Export ZipVoice-Distill ONNX

Examples:
  python egs/zipvoice/run_post_training.py --work-dir .
  python egs/zipvoice/run_post_training.py --steps 3,4,5,6
  # Use smaller distill trainer for stages 4/6
  python egs/zipvoice/run_post_training.py \
      --steps 4,5,6,8 \
      --use-smaller-distill \
      --student-model-config egs/zipvoice/conf/zipvoice_distill_small.json \
      --teacher-model-config exp/pretrained_zipvoice/model.json
  # Continue after run_finetune_serbian.sh (prefix=juznevesti, num_epochs=100)
  
 nohup python egs/zipvoice/run_post_training.py \
      --steps 3,4,5,6 \
      --use-smaller-distill \
      --base-exp-dir exp/zipvoice_ar \
      --base-avg-epoch 26 \
      --dataset custom \
      --tokenizer espeak \
      --lang ar \
      --distill1-exp-dir exp/zipvoice_distill_1stage_juznevesti \
      --distill2-exp-dir exp/zipvoice_distill_juznevesti \
      --token-file data/tokens_arabic_espeak_360.txt \
      --train-manifest data/fbank/arabic_tok_filtered_cuts_train.jsonl.gz \
      --dev-manifest data/fbank/arabic_tok_filtered_cuts_dev.jsonl.gz \
      --model-config exp/pretrained_zipvoice/model.json \
      --student-model-config egs/zipvoice/conf/zipvoice_distill_small.json \
      --teacher-model-config exp/pretrained_zipvoice/model.json \
      --world-size 2 >> run_post_training.txt 2>&1 &

 nohup python egs/zipvoice/run_post_training.py \
      --steps 5,6 \
      --distill1-iters 34000 \
        --distill1-avg 4 \
      --base-exp-dir exp/zipvoice_sr \
      --base-avg-epoch 15 \
      --distill1-exp-dir exp/zipvoice_distill_1stage_juznevesti \
      --distill2-exp-dir exp/zipvoice_distill_juznevesti \
      --dataset custom \
      --tokenizer espeak \
      --lang sr \
      --token-file data/tokens_juznevesti_espeak_360.txt \
      --train-manifest data/fbank/juznevesti_tok_filtered_cuts_train.jsonl.gz \
      --dev-manifest data/fbank/juznevesti_tok_filtered_cuts_dev.jsonl.gz \
      --model-config exp/pretrained_zipvoice/model.json \
      --student-model-config egs/zipvoice/conf/zipvoice_distill_small.json \
      --teacher-model-config exp/pretrained_zipvoice/model.json \
      --world-size 2 >> run_post_training_juznevesti.txt 2>&1 &

      
 nohup python egs/zipvoice/run_post_training.py \
  --steps 5,6 \
  --distill1-iters 35000 \
  --base-exp-dir exp/zipvoice_ar \
  --dataset custom \
  --tokenizer espeak \
  --lang ar \
  --token-file data/tokens_arabic_espeak_360.txt \
  --train-manifest data/fbank/arabic_tok_filtered_cuts_train.jsonl.gz \
  --dev-manifest data/fbank/arabic_tok_filtered_cuts_dev.jsonl.gz \
  --model-config exp/pretrained_zipvoice/model.json \
  --student-model-config egs/zipvoice/conf/zipvoice_distill_small.json \
  --teacher-model-config exp/pretrained_zipvoice/model.json \
  --world-size 2 >> run_post_training.txt 2>&1 &
  
  

 python egs/zipvoice/run_post_training.py \
      --steps 3,4,5,6,7,8 \
      --base-exp-dir exp/zipvoice_sr \
      --base-avg-epoch 100 \
      --dataset custom \
      --tokenizer espeak \
      --lang sr \
      --token-file data/tokens_juznevesti_espeak_360.txt \
      --train-manifest data/fbank/juznevesti_tok_filtered_cuts_train.jsonl.gz \
      --dev-manifest data/fbank/juznevesti_tok_filtered_cuts_dev.jsonl.gz \
      --model-config exp/pretrained_zipvoice/model.json \
      --world-size 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set


ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_steps(steps: Optional[str], stage: int, stop_stage: int) -> Set[int]:
    if steps is None:
        return {s for s in range(stage, stop_stage + 1) if 3 <= s <= 8}

    out: Set[int] = set()
    for item in steps.split(","):
        item = item.strip()
        if not item:
            continue
        s = int(item)
        if s < 3 or s > 8:
            raise ValueError(f"Stage must be between 3 and 8, got {s}")
        out.add(s)
    return out


def run_cmd(cmd: List[str], work_dir: Path, env: dict) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(work_dir), env=env, check=True)


def avg_checkpoint_name(avg: int, epoch: Optional[int], iters: Optional[int]) -> str:
    if epoch is not None:
        return f"epoch-{epoch}-avg-{avg}.pt"
    if iters is not None:
        return f"iter-{iters}-avg-{avg}.pt"
    raise ValueError("Either epoch or iter must be provided for averaging.")


def add_data_args(
    cmd: List[str],
    args: argparse.Namespace,
    include_model_config: bool = True,
) -> None:
    cmd.extend(
        [
            "--dataset",
            args.dataset,
            "--max-duration",
            str(args.max_duration),
            "--min-len",
            str(args.min_len),
            "--max-len",
            str(args.max_len),
            "--num-workers",
            str(args.num_workers),
            "--num-buckets",
            str(args.num_buckets),
            "--tokenizer",
            args.tokenizer,
            "--token-file",
            args.token_file,
            "--lang",
            args.lang,
        ]
    )
    if include_model_config:
        cmd.extend(["--model-config", args.model_config])
    if args.dataset == "custom":
        if not args.train_manifest or not args.dev_manifest:
            raise ValueError(
                "--train-manifest and --dev-manifest are required when --dataset custom"
            )
        cmd.extend(["--train-manifest", args.train_manifest])
        cmd.extend(["--dev-manifest", args.dev_manifest])
    else:
        cmd.extend(["--manifest-dir", args.manifest_dir])


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--work-dir", type=str, default=".")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--stop-stage", type=int, default=8)
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated stages to run, e.g. 3,4,6,8 (overrides --stage/--stop-stage)."
    )
    parser.add_argument("--dry-run", action="store_true")

    # Common/defaults (taken from run_finetune_arabic.sh where applicable)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--use-fp16", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dataset", type=str, default="custom", choices=["emilia", "libritts", "custom"])
    parser.add_argument("--tokenizer", type=str, default="espeak", choices=["emilia", "libritts", "espeak", "simple"])
    parser.add_argument("--lang", type=str, default="sr")
    parser.add_argument("--token-file", type=str, default="data/tokens_juznevesti_espeak_360.txt")
    parser.add_argument("--manifest-dir", type=str, default="data/fbank")
    parser.add_argument("--train-manifest", type=str, default="data/fbank/juznevesti_tok_filtered_cuts_train.jsonl.gz")
    parser.add_argument("--dev-manifest", type=str, default="data/fbank/juznevesti_tok_filtered_cuts_dev.jsonl.gz")
    parser.add_argument("--model-config", type=str, default="exp/pretrained_zipvoice/model.json")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-buckets", type=int, default=60)
    parser.add_argument("--max-duration", type=int, default=80)
    parser.add_argument("--min-len", type=float, default=1.0)
    parser.add_argument("--max-len", type=float, default=30.0)

    # Stage 3 / 7 (base model)
    parser.add_argument("--base-exp-dir", type=str, default="exp/zipvoice_sr")
    parser.add_argument("--base-model-name", type=str, default="zipvoice", choices=["zipvoice"])
    parser.add_argument("--base-avg", type=int, default=4)
    parser.add_argument("--base-avg-epoch", type=int, default=11)
    parser.add_argument("--base-avg-iter", type=int, default=0, help="If > 0, use iter averaging instead of epoch.")

    # Stage 4 / 5 / 6 / 8 (distill)
    parser.add_argument("--distill1-exp-dir", type=str, default="exp/zipvoice_distill_1stage")
    parser.add_argument("--distill1-iters", type=int, default=40000)
    parser.add_argument("--distill1-lr", type=float, default=0.0001)
    parser.add_argument("--distill1-avg", type=int, default=7)
    parser.add_argument(
        "--stage4-teacher-model",
        type=str,
        default="",
        help="Teacher model for stage 4. If empty, use stage-3 averaged checkpoint.",
    )

    parser.add_argument("--distill2-exp-dir", type=str, default="exp/zipvoice_distill")
    parser.add_argument("--distill2-iters", type=int, default=2000)
    parser.add_argument("--distill2-save-every-n", type=int, default=1000)
    parser.add_argument("--distill2-lr", type=float, default=0.0001)
    parser.add_argument(
        "--use-smaller-distill",
        action="store_true",
        help="Use zipvoice.bin.train_zipvoice_distill_smaller for stages 4 and 6.",
    )
    parser.add_argument(
        "--student-model-config",
        type=str,
        default="",
        help="Student config for smaller distill trainer.",
    )
    parser.add_argument(
        "--teacher-model-config",
        type=str,
        default="",
        help="Teacher config for stage 4 when using smaller distill trainer.",
    )
    parser.add_argument(
        "--stage6-teacher-model-config",
        type=str,
        default="",
        help="Teacher config for stage 6 when using smaller distill trainer. "
        "If empty, student config is used.",
    )
    parser.add_argument(
        "--stage6-teacher-model",
        type=str,
        default="",
        help="Teacher model for stage 6. If empty, use stage-5 averaged checkpoint.",
    )
    parser.add_argument(
        "--distill2-checkpoint-name",
        type=str,
        default="",
        help="Checkpoint name for stage 8 ONNX export. Default: checkpoint-{distill2-iters}.pt",
    )

    args = parser.parse_args()
    work_dir = Path(args.work_dir).resolve()
    stages = parse_steps(args.steps, args.stage, args.stop_stage)

    base_avg_epoch: Optional[int] = None
    base_avg_iter: Optional[int] = None
    if args.base_avg_iter > 0:
        base_avg_iter = args.base_avg_iter
    else:
        base_avg_epoch = args.base_avg_epoch

    base_avg_ckpt_name = avg_checkpoint_name(
        avg=args.base_avg, epoch=base_avg_epoch, iters=base_avg_iter
    )
    base_avg_ckpt_path = str(Path(args.base_exp_dir) / base_avg_ckpt_name)

    distill1_avg_ckpt_name = f"iter-{args.distill1_iters}-avg-{args.distill1_avg}.pt"
    distill1_avg_ckpt_path = str(Path(args.distill1_exp_dir) / distill1_avg_ckpt_name)

    stage4_teacher = args.stage4_teacher_model or base_avg_ckpt_path
    stage6_teacher = args.stage6_teacher_model or distill1_avg_ckpt_path

    distill2_export_ckpt = (
        args.distill2_checkpoint_name
        if args.distill2_checkpoint_name
        else f"checkpoint-{args.distill2_iters}.pt"
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROOT_DIR}:{existing_pythonpath}" if existing_pythonpath else str(ROOT_DIR)
    )

    py = sys.executable
    distill_module = (
        "zipvoice.bin.train_zipvoice_distill_smaller"
        if args.use_smaller_distill
        else "zipvoice.bin.train_zipvoice_distill"
    )

    plan = [s for s in [3, 4, 5, 6, 7, 8] if s in stages]
    print(f"Work dir: {work_dir}")
    print(f"Stages to run: {plan}")
    if args.dry_run:
        print("Dry run enabled; commands will only be printed.")

    def run_or_print(cmd: List[str]) -> None:
        if args.dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}", flush=True)
            return
        run_cmd(cmd, work_dir=work_dir, env=env)

    if 3 in stages:
        cmd = [
            py,
            "-m",
            "zipvoice.bin.generate_averaged_model",
            "--avg",
            str(args.base_avg),
            "--model-name",
            args.base_model_name,
            "--exp-dir",
            args.base_exp_dir
        ]
        if base_avg_epoch is not None:
            cmd.extend(["--epoch", str(base_avg_epoch)])
        else:
            cmd.extend(["--iter", str(base_avg_iter)])
        run_or_print(cmd)

    if 4 in stages:
        cmd = [
            py,
            "-m",
            distill_module,
            "--world-size",
            str(args.world_size),
            "--use-fp16",
            str(args.use_fp16),
            "--num-iters",
            str(args.distill1_iters),
            "--base-lr",
            str(args.distill1_lr),
            "--teacher-model",
            stage4_teacher,
            "--distill-stage",
            "first",
            "--exp-dir",
            args.distill1_exp_dir
        ]
        if args.use_smaller_distill:
            add_data_args(cmd, args, include_model_config=False)
            if args.student_model_config:
                cmd.extend(["--student-model-config", args.student_model_config])
            if args.teacher_model_config:
                cmd.extend(["--teacher-model-config", args.teacher_model_config])
            elif args.model_config:
                cmd.extend(["--teacher-model-config", args.model_config])
        else:
            add_data_args(cmd, args, include_model_config=True)
        run_or_print(cmd)

    if 5 in stages:
        cmd = [
            py,
            "-m",
            "zipvoice.bin.generate_averaged_model",
            "--iter",
            str(args.distill1_iters),
            "--avg",
            str(args.distill1_avg),
            "--model-name",
            "zipvoice_distill",
            "--exp-dir",
            args.distill1_exp_dir
        ]
        run_or_print(cmd)

    if 6 in stages:
        cmd = [
            py,
            "-m",
            distill_module,
            "--world-size",
            str(args.world_size),
            "--use-fp16",
            str(args.use_fp16),
            "--num-iters",
            str(args.distill2_iters),
            "--save-every-n",
            str(args.distill2_save_every_n),
            "--base-lr",
            str(args.distill2_lr),
            "--teacher-model",
            stage6_teacher,
            "--distill-stage",
            "second",
            "--exp-dir",
            args.distill2_exp_dir
        ]
        if args.use_smaller_distill:
            add_data_args(cmd, args, include_model_config=False)
            if args.student_model_config:
                cmd.extend(["--student-model-config", args.student_model_config])
            if args.stage6_teacher_model_config:
                cmd.extend(
                    ["--teacher-model-config", args.stage6_teacher_model_config]
                )
            elif args.student_model_config:
                cmd.extend(["--teacher-model-config", args.student_model_config])
        else:
            add_data_args(cmd, args, include_model_config=True)
        run_or_print(cmd)

    if 7 in stages:
        cmd = [
            py,
            "-m",
            "zipvoice.bin.onnx_export",
            "--model-name",
            "zipvoice",
            "--model-dir",
            args.base_exp_dir,
            "--checkpoint-name",
            base_avg_ckpt_name,
            "--onnx-model-dir",
            args.base_exp_dir
        ]
        run_or_print(cmd)

    if 8 in stages:
        cmd = [
            py,
            "-m",
            "zipvoice.bin.onnx_export",
            "--model-name",
            "zipvoice_distill",
            "--model-dir",
            args.distill2_exp_dir,
            "--checkpoint-name",
            distill2_export_ckpt,
            "--onnx-model-dir",
            args.distill2_exp_dir
        ]
        run_or_print(cmd)


if __name__ == "__main__":
    main()
