
# ZipVoice Multilingual Finetuning

<p align="center">
  <a style="text-decoration:none" href="https://huggingface.co/spaces/karim1993/zipvoice-multilingual-tts-demo">
    <img style="margin-right:8px" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  <a style="text-decoration:none" href="https://huggingface.co/karim1993/zipvoice-sr-finetuned">
    <img style="margin-right:8px" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Serbian%20Model-FFD21E" alt="Hugging Face Serbian Model">
  </a>
  <a style="text-decoration:none" href="https://huggingface.co/karim1993/zipvoice-ar-finetuned">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Arabic%20Model-FFD21E" alt="Hugging Face Arabic Model">
  </a>
</p>

Fine-tuning and inference setup for multilingual ZipVoice models, focused on:
- Arabic (`ar`)
- Serbian (`sr`)

Base model used in this project: **ZipVoice**.

## Attribution

- GitHub: https://github.com/k2-fsa/ZipVoice

This repository includes:
- model training/evaluation scripts
- a Gradio inference app (`app.py`)
- Hugging Face Space-compatible setup (`requirements.txt`, `packages.txt`, README front matter)

## Hugging Face Model Links (Quick Access)

Demo page:
- [zipvoice-multilingual-tts-demo](https://huggingface.co/spaces/karim1993/zipvoice-multilingual-tts-demo)

Arabic model page:
- [karim1993/zipvoice-ar-finetuned](https://huggingface.co/karim1993/zipvoice-ar-finetuned)

Serbian model page:
- [karim1993/zipvoice-sr-finetuned](https://huggingface.co/karim1993/zipvoice-sr-finetuned)

## Training Datasets

### Serbian
- CLARIN Dataset `11356/1834`  
  [Dataset link](https://www.clarin.si/repository/xmlui/handle/11356/1834)
- CLARIN Dataset `11356/1679`  
  [Dataset link](https://www.clarin.si/repository/xmlui/handle/11356/1679)

### Arabic
- Common Voice  
  [Dataset link](https://datacollective.mozillafoundation.org/datasets/cmj8u3os6000tnxxb169x1zdc)
- ArVoice (Human split only)  
  [Dataset link](https://huggingface.co/datasets/MBZUAI/ArVoice)
- MGB2 Arabic  
  [Dataset link](https://huggingface.co/datasets/MohamedRashad/mgb2-arabic)

## Evaluation Metrics

### Serbian (`sr`)
| Metric | Count | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| wer | 92 | 0.17 | 0.08 | 0.24 | 0.00 | 1.00 |
| cer | 92 | 0.10 | 0.02 | 0.25 | 0.00 | 2.00 |
| wav_seconds | 92 | 4.18 | 3.63 | 2.92 | 0.24 | 22.98 |
| wavlm_sim | 92 | 0.63 | 0.69 | 0.15 | 0.01 | 0.83 |

### Arabic (`ar`)
| Metric | Count | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| wer | 100 | 0.14 | 0.00 | 0.20 | 0.00 | 1.00 |
| cer | 100 | 0.05 | 0.00 | 0.14 | 0.00 | 1.30 |
| wav_seconds | 100 | 3.39 | 2.66 | 2.20 | 0.62 | 12.48 |
| wavlm_sim | 100 | 0.45 | 0.48 | 0.14 | 0.01 | 0.69 |

Arabic WER caveat:
Whisper does not output Arabic diacritics (علامات التشكيل / الحركات), so diacritic mismatches are not reflected in WER.
`◌َ فَتْحَة | ◌ُ ضَمَّة | ◌ِ كَسْرَة | ◌ْ سُكُون | ◌ّ شَدَّة | ◌ً تَنْوِينُ الفَتْح | ◌ٌ تَنْوِينُ الضَّم | ◌ٍ تَنْوِينُ الكَسْر | ◌ٓ مَدَّة | ◌ٰ أَلِف خَنْجَرِيَّة`

## Training Summary

Reported training time:
- 2 days for each language
- 1 extra day for each distilled model

### Arabic Training Data
- `total_rows`: `153666`
- `total_duration`: `395.46 hours`

### Serbian Training Data
- `total_rows`: `92177`
- `total_duration`: `280.87 hours`

## Gradio Inference

Main app:
- `app.py`

Local run:
```bash
python3 -m pip install -r requirements.txt
python3 -m app
```

System packages required:
- `ffmpeg`
- `espeak-ng`

## Notes For Hugging Face Spaces

- This README includes Spaces metadata in the front matter (`sdk: gradio`, `app_file: app.py`).
- The app is configured for Space deployment and can auto-download model artifacts/source when enabled by environment variables.
- Arabic model page (again): [karim1993/zipvoice-ar-finetuned](https://huggingface.co/karim1993/zipvoice-ar-finetuned)
- Serbian model page (again): [karim1993/zipvoice-sr-finetuned](https://huggingface.co/karim1993/zipvoice-sr-finetuned)
