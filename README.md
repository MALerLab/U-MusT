# U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance Audio

[![Paper](https://img.shields.io/badge/IEEE%20TASLP-10.1109%2FTASLPRO.2025.3648794-blue)](https://doi.org/10.1109/TASLPRO.2025.3648794)
[![arXiv](https://img.shields.io/badge/arXiv-2505.12863-b31b1b)](https://arxiv.org/abs/2505.12863)
[![Demo](https://img.shields.io/badge/Demo-sakem.in%2Fu--must-green)](https://sakem.in/u-must/)

Official implementation of **U-MusT** (IEEE Transactions on Audio, Speech and Language Processing, vol. 34, 2026), by Jongmin Jung*, Dongmin Kim*, Sihun Lee, Seola Cho, Hyungjoon Soh, Irmak Bukey, Chris Donahue, and Dasaem Jeong (*equal contribution).

U-MusT is a unified sequence-to-sequence framework that translates between four modalities of Western music — **score images**, **symbolic notation (MusicXML/LMX)**, **performance MIDI**, and **audio** — with a single Transformer encoder–decoder per translation direction. All modalities are discretized into tokens (RQ-VAE for images, DAC for audio, LMX for notation, MT3-style events for MIDI), enabling joint multi-task training of OMR, AMT, MIDI-to-audio synthesis, and the first musically-coherent direct score-image-to-audio generation.

The companion **YouTube Score Video (YTSV) dataset** (433k image–audio pairs, 1,341 hours) is released separately at [MALerLab/youtube-score-video-dataset](https://github.com/MALerLab/youtube-score-video-dataset).

## Repository structure

```
umust/                  Core package: model, data pipeline, tokenization, trainer
  ├── model_zoo.py        MultimodalTranslator (encoder–decoder + codebook sub-decoder)
  ├── encoders.py         Input encoders (incl. PerceiverTF wrapper)
  ├── decoders.py         Main decoder + sub-decoder for RVQ codebooks
  ├── trainer.py          MultimodalTrainer (multi-task curriculum training)
  ├── data_utils.py       MultimodalTokenDatasetMaker, samplers, collate fns
  ├── vocab_utils.py      Unified vocabulary over all modality token sets
  ├── data_decode_utils.py TensorDecoder: tokens → image / audio / MIDI / MusicXML
  ├── lmx_utils/          Linearized MusicXML (LMX) tokenization (after Mayer et al.)
  ├── midi_utils/         MT3-style MIDI event tokenization (+ dataset preprocessing)
  └── yourmt3plus/        Trimmed vendored modules from YourMT3+ (Chang et al.)
rqvae/                  Vendored & modified RQ-VAE (Kakao Brain, Apache-2.0) for image tokens
config/                 Hydra configs (config_mm.yaml + data / nn_params / wandb groups)
dataset_pair_paths/     Train/valid/test split manifests per dataset
scripts/                Dataset split builders and evaluation scripts
vocab/                  LMX token vocabularies
mxl_render_scripts/     MusicXML → PDF/system-image rendering (MuseScore)
train_multimodal.py     Training entry point
infer.py                MusicXML → audio inference pipeline
```

## Installation

Python 3.10 with CUDA is assumed.

```bash
pip install -r requirements.txt
```

System dependencies for rendering and synthesis:

```bash
apt-get install ffmpeg fluidsynth fluid-soundfont-gm
./install_mscore-3.6.2.sh   # MuseScore 3.6.2 AppImage (LMX/MusicXML rendering)
```

A `Dockerfile` and `docker_run.sh` are provided with all of the above preinstalled.

## Tokenizers

Continuous modalities are discretized by two neural codecs trained on classical music, both with **4 unshared codebooks × 1,024 codes**:

- **Image — RQ-VAE** ([Lee et al. 2022](https://github.com/kakaobrain/rq-vae-transformer)): 16× compression, model dim 256, attention blocks removed, grayscale, resolution-adaptive multi-height training. Our modifications live in the vendored `rqvae/` package (dataset, trainer, LPIPS loss); training uses the original repo's stage-1 driver with these modules.
- **Audio — DAC** ([Kumar et al. 2023](https://github.com/descriptinc/descript-audio-codec)): retrained with 4 codebooks, hop size 512 at 44.1 kHz mono (≈86 token sets/s), following the official training recipe.

Pretrained tokenizer checkpoints are loaded from local directories: place the RQ-VAE checkpoint (`config.yaml` + `*.pt`) under `vq_models/<model_string>/` (e.g. `vq_models/unirqvae_f16_c1024_k4/`) and the DAC checkpoint (`weights.pth`) under `dac_models/<name>/` (e.g. `dac_models/unidac4/`). The base directories can be overridden with `+data.vq_model_dir=<path>` and `+data.dac_model_dir=<path>`.

## Data preparation

Datasets used in the paper: YTSV, GrandStaff, OLiMPiC, MAESTRO, MusicNet (with MusicNetEM labels), SLakh, and BPSD (test only).

1. Download each dataset and place it under a common root (e.g. `dataset/`).
2. Preprocess MIDI-audio datasets with `umust/midi_utils/preprocess/preprocess_{maestro,musicnet,slakh,asap}.py`.
3. Build or reuse split manifests in `dataset_pair_paths/` (`scripts/make_*_split*.py`). Manifests for the public datasets are included; YTSV manifests are built with `scripts/make_youtube_split_json.py` after processing YTSV with the [dataset pipeline](https://github.com/MALerLab/youtube-score-video-dataset).

Score images are normalized with fine-tuned YOLOv8 models (system detection + staff-height detection, staff height 18 px) as described in Appendix A-D of the paper.

## Training

Each translation direction is trained as a separate model with the same architecture (12+12 layers, dim 1024, 16 heads, ~600k steps on 2× H100):

```bash
# Image-to-Audio direction (OMR + MIDI-to-audio + image-to-audio)
python3 train_multimodal.py --config-name=config_mm \
  data=multimodal_omr_direction_yolo data.data_dir=dataset/ train_params.world_size=2

# Audio-to-Image direction (AMT + LMX-to-image + audio-to-image)
python3 train_multimodal.py --config-name=config_mm \
  data=multimodal_amt_direction data.data_dir=dataset/ train_params.world_size=2
```

Task curriculum (which tasks enter the batch mixture at which step) and dataset sampling weights are defined in the `config/data/*.yaml` recipes. Per-task fine-tuning (50k steps, lr 1e-5) is configured through `finetune_params`.

Logging uses [Weights & Biases](https://wandb.ai); set your entity/project in `config/wandb_config/default.yaml` or disable with `general.make_log=False`.

## Evaluation

```bash
# OMR symbol error rate on OLiMPiC (scanned + synthetic)
python3 scripts/test_olimpic.py --run_path <run_dir> --data_dir dataset/

# AMT note-onset F1 on MAESTRO / MusicNet / SLakh
python3 scripts/test_amt.py --run_path <run_dir> --target_dataset musicnet --data_dir dataset/
```

`<run_dir>` is a training run directory containing `files/config.yaml` and `files/checkpoints/`.

## Inference (MusicXML → audio)

```bash
python3 infer.py input.mxl --instrument piano -o output/ --models_dir models/
```

Renders the score with MuseScore, tokenizes each system image, and generates performance audio with the I2A model — or via Docker: `./docker_run.sh input.mxl output/`.

Model checkpoints for the translation models are not included in this repository.

## Acknowledgments

This repository vendors adapted code from:
- [rq-vae-transformer](https://github.com/kakaobrain/rq-vae-transformer) (Kakao Brain, Apache-2.0) — `rqvae/`
- [YourMT3+](https://github.com/mimbres/YourMT3) (Chang et al.) — `umust/yourmt3plus/`, MIDI event tokenization in `umust/midi_utils/`
- [Linearized MusicXML / Olimpic-ICDAR24](https://github.com/ufal/olimpic-icdar24) (Mayer et al.) — `umust/lmx_utils/`

## Citation

```bibtex
@article{jung2026umust,
  title   = {U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance Audio},
  author  = {Jung, Jongmin and Kim, Dongmin and Lee, Sihun and Cho, Seola and Soh, Hyungjoon and Bukey, Irmak and Donahue, Chris and Jeong, Dasaem},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  volume  = {34},
  pages   = {1876--1891},
  year    = {2026},
  doi     = {10.1109/TASLPRO.2025.3648794}
}
```

## License

MIT (see [LICENSE](LICENSE)). Vendored third-party code retains its original license and attribution.
