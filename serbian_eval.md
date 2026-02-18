# Evaluation Report

_Generated: 2026-02-17T19:42:17_

## Config
| Key | Value |
| --- | --- |
| asr_language | hr |
| asr_model_id | openai/whisper-large-v3-turbo |
| checkpoint_path | exp/zipvoice_sr/epoch-15-avg-4.pt |
| dataset_tsv | data/raw/juznevesti_dev.tsv |
| espeak_lang | sr |
| lang | sr |
| max_samples | None |
| model_config_path | exp/zipvoice_sr/model.json |
| model_type | zipvoice |
| prompt_mode | self |
| sim_checkpoint | wavlm_large_finetune.pth |
| sim_model_name | wavlm_large |
| sim_wavlm_ckpt | wavlm_large.pt |
| sim_wavlm_ckpt_used | wavlm_large.pt |
| token_file | exp/zipvoice_sr/tokens.txt |
| tokenizer | espeak |

## Counts
| Metric | Value |
| --- | --- |
| total | 92 |
| ok | 92 |
| error | 0 |


## Per-Language Metrics
### `sr`
| Metric | Count | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| wer | 92 | 0.17 | 0.08 | 0.24 | 0.00 | 1.00 |
| cer | 92 | 0.10 | 0.02 | 0.25 | 0.00 | 2.00 |
| wav_seconds | 92 | 4.18 | 3.63 | 2.92 | 0.24 | 22.98 |
| wavlm_sim | 92 | 0.63 | 0.69 | 0.15 | 0.01 | 0.83 |
