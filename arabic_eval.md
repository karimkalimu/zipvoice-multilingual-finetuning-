# Evaluation Report

_Generated: 2026-02-17T19:45:19_

## Config
| Key | Value |
| --- | --- |
| asr_language | ar |
| asr_model_id | openai/whisper-large-v3-turbo |
| checkpoint_path | exp/zipvoice_ar/epoch-26-avg-4.pt |
| dataset_tsv | data/raw/arabic_dev.tsv |
| espeak_lang | ar |
| generated_dir | eval_outputs/generated_wavs |
| lang | ar |
| model_config_path | exp/pretrained_zipvoice/model.json |
| model_type | zipvoice |
| prompt_mode | self |
| sim_checkpoint | wavlm_large_finetune.pth |
| sim_model_name | wavlm_large |
| sim_wavlm_ckpt | wavlm_large.pt |
| sim_wavlm_ckpt_used | wavlm_large.pt |
| token_file | data/tokens_arabic_espeak_360.txt |
| tokenizer | espeak |

## Counts
| Metric | Value |
| --- | --- |
| total | 100 |
| ok | 100 |
| error | 0 |


## Per-Language Metrics
### `ar`
| Metric | Count | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| wer | 100 | 0.14 | 0.00 | 0.20 | 0.00 | 1.00 |
| cer | 100 | 0.05 | 0.00 | 0.14 | 0.00 | 1.30 |
| wav_seconds | 100 | 3.39 | 2.66 | 2.20 | 0.62 | 12.48 |
| wavlm_sim | 100 | 0.45 | 0.48 | 0.14 | 0.01 | 0.69 |
