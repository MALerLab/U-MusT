# U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance Audio

[![Paper](https://img.shields.io/badge/IEEE%20TASLP-10.1109%2FTASLPRO.2025.3648794-blue)](https://doi.org/10.1109/TASLPRO.2025.3648794)
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11316398-00629B)](https://ieeexplore.ieee.org/document/11316398)
[![Demo](https://img.shields.io/badge/Demo-sakem.in%2Fu--must-green)](https://sakem.in/u-must/)

Official implementation of
> **U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance Audio**<br>
> Jongmin Jung\*, Dongmin Kim\*, Sihun Lee, Seola Cho, Hyungjoon Soh, Irmak Bukey, Chris Donahue, and Dasaem Jeong (\*equal contribution)<br>
> *IEEE Transactions on Audio, Speech and Language Processing, vol. 34, 2026*

Music exists as score images, symbolic notation, MIDI, and audio, and translating between those modalities covers much of music information retrieval — optical music recognition, automatic music transcription, synthesis. Past work builds a specialized model per task. **U-MusT** instead discretizes all four modalities into one token vocabulary, so a standard encoder–decoder Transformer can treat every translation as the same sequence-to-sequence problem and learn them jointly. Two things make that viable: the **YouTube Score Video (YTSV)** dataset, 1,341 hours of paired score-image and audio data that is an order of magnitude larger than any prior music modal-translation corpus, and a unified tokenization scheme. Joint training beats single-task baselines across the board — optical music recognition on scanned scores drops from 24.58% to 13.67% symbol error rate — and produces the first musically-coherent direct score-image-to-audio generation. This repository is the training, evaluation, and inference code; the dataset lives in its own repository.

## What's in this release

Everything below ships in this repository unless the Location column says otherwise.

| Component | What it is | Location |
|---|---|---|
| Translation models | Encoder–decoder Transformer, one per direction (I2A, A2I) | `umust/`, `config/` |
| Training entry point | Hydra-configured multi-task trainer | `train_multimodal.py` |
| Evaluation | OMR symbol error rate, AMT note-onset F1 | `scripts/test_olimpic.py`, `scripts/test_amt.py` |
| Inference | MusicXML → performance audio, end to end | `infer.py` |
| Tokenizer checkpoints | RQ-VAE (score images), DAC (audio) | `vq_models/`, `dac_models/` |
| Score-layout detectors | YOLO system + staff-height models | [MALerLab/ls-yolo releases](https://github.com/MALerLab/ls-yolo/releases) (auto-downloaded) |
| Dataset split manifests | Every train/valid/test split used in the paper | `dataset_pair_paths/` |
| Token-baking scripts | Reproduce the image/audio token datasets | `scripts/bake_image_tokens.py`, `scripts/bake_audio_tokens.py` |
| **YTSV dataset** | **1,341 h of paired score-image/audio — the paper's main dataset** | **[MALerLab/youtube-score-video-dataset](https://github.com/MALerLab/youtube-score-video-dataset)** |
| Translation-model weights | Three checkpoints — I2A piano, I2A strings, A2I | [malerlab/u-must](https://huggingface.co/malerlab/u-must) (gated) |
| Tokenized datasets | Image and audio tokens for every corpus except GrandStaff | [Hugging Face](#released-weights-and-data) |

## How the pieces fit together

![Overview of U-MusT. Two directions, each a Transformer encoder-decoder. Image-to-Audio takes score images through an RQVAE encoder or MIDI through a MIDI tokenizer, and emits audio tokens decoded by DAC or LMX tokens decoded to notation. Audio-to-Image is the reverse, taking audio through a DAC encoder or notation through an LMX tokenizer and emitting image tokens decoded by RQVAE or MIDI tokens.](figures/umust_overview.jpg)

<sub>Figure 2 from the paper. &copy; 2025 IEEE.</sub>

Each modality is discretized by its own tokenizer — **RQ-VAE** for score images, **DAC** for audio, **LMX** for notation, **MT3**-style events for MIDI — into one shared vocabulary. Notation and MIDI are already discrete and need no learned codec. Two separately trained models with identical architecture then run in opposite directions: the **I2A** model takes image or MIDI tokens and emits LMX or audio tokens, covering OMR, image-to-audio, and MIDI-to-audio synthesis; the **A2I** model takes audio or LMX tokens and emits image or MIDI tokens, covering AMT, audio-to-image, and notation-to-image. The two directions do not share weights.

## Getting started

Pick the path that matches what you want to do. All three assume [Installation](#installation) is done first.

### (a) I just want to turn a score into audio

Request access to the weights at [malerlab/u-must](https://huggingface.co/malerlab/u-must), then download them and the codecs:

```bash
hf download malerlab/u-must --local-dir models/
hf download malerlab/unirqvae3-ytsv --local-dir vq_models/unirqvae3_f16_c1024_k4/
hf download malerlab/unidac4-ytsv  --local-dir dac_models/unidac4/
python3 infer.py input.mxl --run_path models/run-20250225_062905-9n1554as -o output/
```

`--run_path` also takes any run directory you trained yourself (path (c)).

`<run_dir>` is a training run directory containing `files/config.yaml` and `files/checkpoints/*.pt`. The script renders the score with MuseScore, detects and crops each musical system with YOLO, tokenizes the crops with the RQ-VAE, generates audio tokens, and decodes them with DAC to `output/final_output.wav`. The YOLO weights download automatically on first run.

### (b) I want to reproduce the paper's numbers

```bash
# OMR symbol error rate on OLiMPiC (synthetic + scanned)
python3 scripts/test_olimpic.py --run_path <run_dir> --data_dir dataset/

# AMT note-onset F1 on MusicNet / MAESTRO
python3 scripts/test_amt.py --run_path <run_dir> --target_dataset musicnet --data_dir dataset/
```

See [Data preparation](#data-preparation) for what has to be on disk, and [Results at a glance](#results-at-a-glance) for the targets.

### (c) I want to train from scratch

```bash
python3 train_multimodal.py --config-name=config_mm \
  data=omr_piano_synth_long data.data_dir=dataset/ train_params.world_size=2
```

See [Training](#training) for the full recipe list and the paper's settings.

## Datasets at a glance

The paper's own dataset contribution is **YTSV**, released separately at [MALerLab/youtube-score-video-dataset](https://github.com/MALerLab/youtube-score-video-dataset).

- **433,920 image–audio pairs from 12,217 score-following YouTube videos, 1,341 hours in total**, covering roughly 10,000 unique pieces by more than 2,000 composers.
- 13 instrumentation categories; Piano Solo dominates at 9,052 videos and 762 hours. The piano subset used to train the released I2A model (**YTSV-P**) is 252k segments and 815 hours.
- The videos themselves are **not redistributed**. That repository publishes the metadata (including video links) and the preprocessing pipeline — slide segmentation and YOLOv8 system cropping — and you download the media yourself.
- Its metadata is licensed **CC BY-NC-SA 4.0** and its code MIT. The authors ask that dependent work cite the U-MusT paper; see [Citation](#citation).

The full training mixture combines YTSV with public corpora. `N` is aligned entries and `H` is audio hours; `△` marks MIDI derived from audio-aligned scores rather than captured directly, and `*` marks hours estimated for notation-only corpora (both per the paper's Table II).

| Subset | Image | MusicXML | MIDI | Audio | N | H |
|---|---|---|---|---|---|---|
| YTSV | ✓ | – | – | ✓ | 433,920 | 1,341 |
| GrandStaff | ✓ | ✓ | – | – | 7,661 | 23\* |
| OLiMPiC | ✓ | ✓ | – | – | 17,945 | 47\* |
| MusicNet_EM | – | – | △ | ✓ | 330 | 33 |
| MAESTRO | – | – | ✓ | ✓ | 1,276 | 199 |
| SLakh | – | – | ✓ | ✓ | 2,100 | 145 |
| BPSD | ✓ | ✓ | △ | ✓ | 32 | 14 |

**BPSD** (Beethoven Piano Sonata Dataset) is **test-only** and the only subset carrying all four modalities. It ships no system-level image alignment, so the authors hand-annotated a 9-piece test subset drawn from the MAESTRO test split and excluded those pieces from GrandStaff training. SLakh is used in training but excluded from reported results, since the paper's focus is Western classical music.

## Results at a glance

Optical music recognition, symbol error rate on LMX token sequences — lower is better (paper Table VII). Rows are cumulative ablations of the I2A model; **Zeus** is the authors' retraining of the prior state-of-the-art pianoform OMR system on the identical split.

| Method | OLiMPiC synthetic | OLiMPiC scanned | BPSD scanned |
|---|---|---|---|
| OMR only | 15.90 | 24.58 | 45.39 |
| + Image-to-Audio | 10.57 | 15.45 | 23.85 |
| + MIDI-to-Audio (full) | **9.72** | **13.67** | **23.36** |
| Zeus (prior SOTA, retrained) | 10.10 | 14.45 | 31.24 |

The abstract's headline 24.58 → 13.67 is the **OLiMPiC scanned** column, comparing the paper's own single-task OMR baseline against its full multi-task model. The full model also beats the external state of the art on that column (13.67 vs 14.45) and by a wide margin on BPSD (23.36 vs 31.24).

Automatic music transcription, note-onset F1 — higher is better (paper Table VIII).

| Method | MusicNet_EM strings | MusicNet_EM woodwinds | MAESTRO |
|---|---|---|---|
| AMT only | 87.21 | 72.04 | 89.40 |
| + Audio-to-Image | **87.28** | 72.61 | 89.38 |
| + LMX-to-Image (full) | 87.25 | **75.52** | **89.45** |

Direct image-to-audio generation, audio-to-image accuracy, MIDI-to-audio synthesis, and the mean-opinion-score listening test are in paper Tables IV–VI; audio examples are on the [demo page](https://sakem.in/u-must/).

## Model at a glance

Both directions use an identical configuration — a sequence-to-sequence Transformer with **12 encoder and 12 decoder layers**, model dimension **1024**, feed-forward size **4096** with GELU, and **16 attention heads**. A one-layer sub-decoder with **8 heads** predicts the four codebook entries within each timestep. Image and audio token embeddings are initialized from the learned RQ-VAE and DAC codebooks; every input token additionally carries a modality-specific positional embedding and a target-modality embedding.

Training uses **AdamW**, initial learning rate **1e-4** with cosine decay to **1e-5** and 2,000 linear warm-up steps, **600,000 updates** at a total batch size of **24** sequence pairs on 2 GPUs, with weighted dataset sampling so no single corpus dominates. Per-task fine-tuning runs a further 50k steps at 1e-5. Tasks enter the mixture on a curriculum: for I2A, OMR from step 0, MIDI-to-audio at 15,000, direct image-to-audio at 50,000; for A2I, AMT from step 0, LMX-to-image at 40,000, audio-to-image at 70,000.

Tokenization, per modality:

- **Score images** — an RQ-VAE retrained from scratch on grayscale sheet music at 16× compression (4 codebooks of 1024 entries, model dimension 256), so each token covers a 16×16 pixel patch. Attention blocks are removed to specialize it for local features. Each musical system becomes a code grid flattened in vertical reading order, with systems joined by a `[SEP]` token so variable-height systems need no padding.
- **Audio** — a DAC retrained on classical music with 4 codebooks instead of the public model's nine, at 44.1 kHz mono with hop 512, giving roughly 86 token sets per second.
- **Notation** — Linearized MusicXML (LMX), a flat single-stream format; codebook positions 2–4 are filled with `[PAD]`.
- **MIDI** — MT3-style event tokens (instrument, pitch, note on/off, 10 ms time markers) via the YourMT3+ implementation, also single-stream.

Because a one-pixel or one-sample offset changes token assignments entirely, training data is augmented over discretization shifts: 32 image variants (8 horizontal × 4 vertical) and 9 audio variants. Details are in the supplementary material at the [article DOI](https://doi.org/10.1109/TASLPRO.2025.3648794).

## Installation

Python 3.10 or newer is required (the data pipeline uses `match`).

```bash
pip install -r requirements.txt
./setup.sh
```

`setup.sh` installs the system packages the pipeline shells out to (`ffmpeg`, `fluidsynth` with the FluidR3 GM soundfont, `xvfb`, and the GL/NSS libraries MuseScore needs), links the soundfont where `midi2audio` expects it, downloads and extracts [**MuseScore 3.6.2**](https://github.com/musescore/MuseScore/releases/tag/v3.6.2), and pre-fetches the YOLO detectors. MuseScore is pinned to 3.6.2 because later versions change score layout, which shifts system cropping.

Two notes on PyTorch. `requirements.txt` points at the cu121 wheel index and constrains `torch>=2.2`, which is required by the `torch.compiler` API and by `einx`'s backend gate. On newer GPUs (Blackwell / sm_120) install PyTorch from the cu128 index **first** — the floor constraint is then already satisfied and `pip install -r requirements.txt` will leave it alone.

## Repository structure

```
umust/                     core package
  model_zoo.py             MultimodalTranslator and single-task models
  encoders.py, decoders.py encoder / decoder with codebook-wise sub-decoder
  trainer.py               multi-task training loop, curriculum, validation
  data_utils.py            token datasets, manifests, collation
  vocab_utils.py           unified vocabulary and index handling
  data_decode_utils.py     decode tokens back to images / MIDI / audio
  lmx_utils/               Linearized MusicXML (vendored, Mayer et al.)
  midi_utils/              MIDI tokenization and dataset preprocessing
  yourmt3plus/             MT3-style MIDI tokenizer (vendored, YourMT3+)
rqvae/                     RQ-VAE image tokenizer (vendored, Kakao Brain)
config/                    Hydra configs
  config_mm.yaml           top-level defaults
  data/                    training recipes (see Training)
  nn_params/               model / encoder / decoder architecture
scripts/                   evaluation, split builders, token baking
dataset_pair_paths/        train/valid/test manifests for every corpus
vq_models/, dac_models/    tokenizer checkpoints
vocab/                     LMX vocabularies
yolo/                      system + staff-height detectors
models/                    expected location for translation-model runs
train_multimodal.py        training entry point
infer.py                   MusicXML -> audio inference
setup.sh                   system dependencies, MuseScore, YOLO weights
```

## Checkpoints

### Released weights and data

The translation weights are **gated**: publicly listed, with access granted on request so that the non-commercial research terms are acknowledged. Everything else is open. The YOLO detectors are not on the Hub — they download automatically from [MALerLab/ls-yolo releases](https://github.com/MALerLab/ls-yolo/releases) on first use.

| Hugging Face repo | What | Access | License |
|---|---|---|---|
| [malerlab/u-must](https://huggingface.co/malerlab/u-must) | translation weights, all three runs | gated | CC BY-NC-SA 4.0 |
| [malerlab/unirqvae3-ytsv](https://huggingface.co/malerlab/unirqvae3-ytsv) | score-image codec, paper results | public | CC BY-NC-SA 4.0 |
| [malerlab/unirqvae-ytsv](https://huggingface.co/malerlab/unirqvae-ytsv) | score-image codec, earlier generation | public | CC BY-NC-SA 4.0 |
| [malerlab/unidac4-ytsv](https://huggingface.co/malerlab/unidac4-ytsv) | audio codec | public | CC BY-NC-SA 4.0 |
| [malerlab/ytsv-unirqvae3-ytsv](https://huggingface.co/datasets/malerlab/ytsv-unirqvae3-ytsv) | YTSV image tokens | gated | CC BY-NC-SA 4.0 |
| [malerlab/ytsv-unidac4-ytsv](https://huggingface.co/datasets/malerlab/ytsv-unidac4-ytsv) | YTSV audio tokens | gated | CC BY-NC-SA 4.0 |
| [malerlab/olimpic-unirqvae3-ytsv](https://huggingface.co/datasets/malerlab/olimpic-unirqvae3-ytsv) | OLiMPiC image + notation tokens | public | **CC BY-SA 4.0** |
| [malerlab/maestro-unidac4-ytsv](https://huggingface.co/datasets/malerlab/maestro-unidac4-ytsv) | MAESTRO + ASAP audio/MIDI tokens | public | CC BY-NC-SA 4.0 |
| [malerlab/musicnet-unidac4-ytsv](https://huggingface.co/datasets/malerlab/musicnet-unidac4-ytsv) | MusicNet + MusicNetEM tokens | public | CC BY-NC-SA 4.0 |
| [malerlab/slakh-unidac4-ytsv](https://huggingface.co/datasets/malerlab/slakh-unidac4-ytsv) | SLakh audio/MIDI tokens | public | CC BY-NC-SA 4.0 |
| [malerlab/bpsd-unirqvae3-unidac4-ytsv](https://huggingface.co/datasets/malerlab/bpsd-unirqvae3-unidac4-ytsv) | BPSD tokens, all four modalities | public | CC BY-NC-SA 4.0 |
| [malerlab/grandstaff-lmx-unirqvae3-ytsv](https://huggingface.co/datasets/malerlab/grandstaff-lmx-unirqvae3-ytsv) | GrandStaff image + notation tokens | public | CC BY-NC-SA 4.0 |

GrandStaff tokens carry an unresolved upstream licensing position; read [Known issues](#known-issues) before relying on them. The GrandStaff and YTSV token repositories are sharded as one gzipped tar per collection group, because the uncompressed form runs to over a million small files — each dataset card documents the extraction.

Licenses differ per repository because each follows the corpus it derives from; see [License summary](#license-summary).

Every script that loads a translation model takes a run directory laid out as `<run_dir>/files/config.yaml` plus `<run_dir>/files/checkpoints/*.pt`, which is what `train_multimodal.py` writes. Pass it with `--run_path` to `infer.py` and the evaluation scripts. `infer.py` also accepts `--instrument {piano,strings}` with `--models_dir`, which resolves a fixed run-directory name under that parent; `--run_path` takes precedence and is the right flag for a model you trained yourself. The tokenizer is read from the run's own `config.data.vq_model`, so runs of either tokenizer generation load correctly.

## Data preparation

Datasets go under one root, passed as `data.data_dir`. Split manifests live in `dataset_pair_paths/` and are referenced by name from each recipe; paths inside them are relative to `data.data_dir`.

1. **YTSV** — get the metadata and preprocessing pipeline from [MALerLab/youtube-score-video-dataset](https://github.com/MALerLab/youtube-score-video-dataset), download the videos listed in `metadata/ytsv_metadata.csv`, and run its segmentation and system-cropping pipeline. Shipped splits: `lsyt.json.gz` (all instrumentation) and `lsyt_piano.json.gz` (piano subset).
2. **GrandStaff, OLiMPiC, MAESTRO, MusicNet (with MusicNetEM labels), SLakh, BPSD** — obtain from their original sources; the builders in `scripts/make_*_split_json.py` regenerate the manifests if you need to change a split.
3. **Bake tokens** — `scripts/bake_image_tokens.py` and `scripts/bake_audio_tokens.py` convert score images and audio into the RQ-VAE and DAC token arrays the loaders expect, including the discretization-shift augmentations.

## Training

```bash
# Image-to-Audio direction, piano (the paper's released I2A configuration)
python3 train_multimodal.py --config-name=config_mm \
  data=omr_piano_synth_long data.data_dir=dataset/ train_params.world_size=2

# Image-to-Audio direction, all instrumentation
python3 train_multimodal.py --config-name=config_mm \
  data=omr_direction_all data.data_dir=dataset/ train_params.world_size=2

# Audio-to-Image direction
python3 train_multimodal.py --config-name=config_mm \
  data=multimodal_amt_direction data.data_dir=dataset/ train_params.world_size=2
```

The shipped recipes, all using the `unirqvae3` image tokenizer and `unidac4` audio tokenizer:

| Recipe | Direction | Input → output modalities |
|---|---|---|
| `omr_piano_synth_long` | I2A, piano | image, MIDI → notation, audio |
| `omr_direction_all` | I2A, all instrumentation | image, MIDI → notation, audio |
| `multimodal_amt_direction` | A2I | notation, audio → image, MIDI |
| `multimodal_trans` | bidirectional (experimental) | all four → all four |
| `finetune_omr` | single-task OMR | image → notation |
| `finetune_m2d` | single-task MIDI-to-audio | MIDI → audio |

Each recipe defines its dataset mixture, sampling weights, curriculum start steps, and sequence-length caps. Paper settings are `train_params.world_size=2` with per-GPU batch 12 (total 24) for 600k steps; the defaults match. Per-task fine-tuning is enabled with `finetune_params.finetune=True finetune_params.finetune_path=<run_dir>/files train_params.initial_lr=1e-5` — note the path points at the `files/` subdirectory, not the run directory itself.

Logging uses [Weights & Biases](https://wandb.ai); set your entity and project in `config/wandb_config/default.yaml`, or disable with `general.make_log=False`. `general.infer_and_log=True` additionally decodes validation predictions to score images, MIDI, and audio — the paper's runs had this on. Validation iterates every dataset in the recipe from the first cycle, because the curriculum gates only the training mixture, so all referenced validation data must be present from the start.

## Known issues

- `multimodal_trans` is the Hydra default recipe but sets `modal_direction: omr`, so it never reaches the bidirectional code path despite its name and modality lists. Pass an explicit `data=` override rather than relying on the default.
- **Two of the three released checkpoints use the earlier `unirqvae` image tokenizer, while the published image-token datasets are `unirqvae3`.** The piano I2A run (`run-20250225_062905-9n1554as`) declares `unirqvae3` and pairs with the published tokens; the strings I2A run (`run-20250130_150202-x9znhap2`) and the A2I run (`run-20250128_025927-ks0ibl4v`) both declare `unirqvae`. Image tokens are not interchangeable between codec generations — the codebooks differ, so the token indices mean different things. Feeding `unirqvae3` tokens to either of those two checkpoints, or evaluating the A2I model's image-token output against them, produces meaningless results. `infer.py` reads the tokenizer from each run's own config, so inference on a downloaded checkpoint selects correctly; the mismatch only bites when pairing a checkpoint with a token dataset by hand.
- `omr_direction_all` and `multimodal_amt_direction` now specify `unirqvae3`, matching the shipped manifests. Retraining from either therefore produces a `unirqvae3`-generation model rather than a reproduction of the released strings or A2I checkpoint. To reproduce those, re-bake image tokens with the `unirqvae` codec using `scripts/bake_image_tokens.py` and set `vq_model: unirqvae` in the recipe.
- `dataset_pair_paths/` ships `asap.json`, `lsyt_multiinst_test.json`, and `lsyt_piano_test_segments.json`, which no shipped recipe references.
- BPSD is used for evaluation in the paper but no shipped evaluation script targets it directly.
- `+data.dac_model_dir` overrides the DAC location for training only; the decode path resolves `dac_models/` relative to the working directory.

## Citation

If you use this code, please cite:

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

**If you use the YTSV dataset**, the same paper is the reference its maintainers ask you to cite, so the entry above covers it. Please also observe the CC BY-NC-SA 4.0 terms on its metadata.

**If you use the vendored or retrained components**, please cite their original work: the RQ-VAE image tokenizer ([Lee et al., rq-vae-transformer](https://github.com/kakaobrain/rq-vae-transformer)), the DAC audio codec ([Kumar et al., descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)), the MT3-style MIDI tokenizer ([Chang et al., YourMT3](https://github.com/mimbres/YourMT3)), and Linearized MusicXML ([Mayer et al., olimpic-icdar24](https://github.com/ufal/olimpic-icdar24)).

## License summary

### Code

| Component | License |
|---|---|
| This repository | MIT |
| — `rqvae/` (Kakao Brain) | Apache-2.0 |
| — `umust/yourmt3plus/` (YourMT3+) | Apache-2.0, see `umust/yourmt3plus/LICENSE` for provenance |
| — `umust/lmx_utils/` (© 2024 Jiří Mayer) | MIT |

### Datasets

**This repository redistributes no corpus audio, images, or scores.** It ships split manifests — lists of relative file paths — and the code to tokenize data you obtain yourself from each corpus's original source. The table records each corpus's license as stated by its publisher, so you can establish your own obligations before using or redistributing anything derived from it.

| Corpus | License as published | Where that is stated |
|---|---|---|
| YTSV metadata | CC BY-NC-SA 4.0 | [youtube-score-video-dataset](https://github.com/MALerLab/youtube-score-video-dataset) |
| MAESTRO v3.0.0 | CC BY-NC-SA 4.0 | [magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro) |
| ASAP | CC BY-NC-SA 4.0 | [fosfrancesco/asap-dataset](https://github.com/fosfrancesco/asap-dataset) `LICENSE.md` |
| MusicNet | CC BY 4.0 | [Zenodo 5120004](https://zenodo.org/records/5120004) |
| MusicNetEM (labels) | CC BY-NC-SA 4.0 | [benadar293.github.io](https://github.com/benadar293/benadar293.github.io) `LICENSE.md` |
| SLakh2100 | CC BY 4.0 | [Zenodo 4599666](https://zenodo.org/records/4599666) |
| Lakh MIDI (SLakh upstream) | CC BY 4.0 | [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/) |
| BPSD v2 | CC BY 3.0 | [Zenodo 12783403](https://zenodo.org/records/12783403) |
| OLiMPiC | CC BY-SA 4.0 | [ufal/olimpic-icdar24](https://github.com/ufal/olimpic-icdar24) |
| OpenScore Lieder (OLiMPiC upstream) | CC0 1.0 | [OpenScore/Lieder](https://github.com/OpenScore/Lieder) |
| GrandStaff-LMX | CC BY-SA 4.0 | [LINDAT 11234/1-5423](http://hdl.handle.net/11234/1-5423) |
| GrandStaff (score images) | **not stated by its publisher**; our token release applies CC BY-NC-SA 4.0 | — |
| KernScores source editions | CC BY-NC-SA 4.0 on some, **unstated on others** | `craigsapp/*` repository `LICENSE.txt` files |

### Known license conflicts

These are unresolved at the time of release. They are recorded rather than papered over, because anyone tokenizing or redistributing this data inherits them.

- **GrandStaff-derived data has no license under which it can be redistributed.** This was investigated against primary sources and the finding is not merely that the chain is ambiguous, but that no grant exists. The GrandStaff archive itself contains no license, readme, copyright or citation file among any of its 234,715 entries, and neither its download location nor its project pages state terms. It decomposes into seven corpora that map one-to-one onto Craig Sapp's KernScores editions, of which only three carry a grant:

  | GrandStaff subtree | Upstream repository | License |
  |---|---|---|
  | `mozart/piano-sonatas` | `craigsapp/mozart-piano-sonatas` | CC BY-NC-SA 4.0 |
  | `joplin/joplin` | `craigsapp/joplin` | CC BY-NC-SA 4.0 |
  | `scarlatti-d/keyboard-sonatas` | `craigsapp/scarlatti-keyboard-sonatas` | CC BY-NC-SA 4.0 |
  | `beethoven/piano-sonatas` | `craigsapp/beethoven-piano-sonatas` | **none** |
  | `chopin/mazurkas` | `craigsapp/chopin-mazurkas` | **none** |
  | `chopin/preludes` | `craigsapp/chopin-preludes` | **none**, copyright asserted |
  | `hummel/preludes` | `craigsapp/hummel-preludes` | **none**, copyright asserted |

  Four subtrees therefore fall under default copyright, two of them with an explicit `!!!YEC: Copyright 2008 by Craig Stuart Sapp` record, and absence of a license is not permission. The three that are licensed are non-commercial and require attribution which GrandStaff has already discarded — it strips every Humdrum `!!!` reference record, so no composer, editor or license notice survives into the distributed data. Separately, GrandStaff-LMX is published on LINDAT as CC BY-SA 4.0 with no recorded analysis of the upstream terms, and CC BY-SA 4.0 is incompatible with CC BY-NC-SA 4.0 in any case. A `license: mit` tag exists on a Hugging Face mirror, but on an empty card with no license file, and the same group tags its other datasets `cc-by-nc-4.0` deliberately.

  GrandStaff tokens **are** published, at [malerlab/grandstaff-lmx-unirqvae3-ytsv](https://huggingface.co/datasets/malerlab/grandstaff-lmx-unirqvae3-ytsv), under CC BY-NC-SA 4.0 — chosen as the strictest of the terms actually stated upstream rather than as a grant that can be fully substantiated. That release reproduces the per-subtree attribution to Craig Stuart Sapp's editions which GrandStaff itself discards, and carries a takedown offer for rights holders. Two notes on the reasoning: GrandStaff-LMX's CC BY-SA 4.0 was applied with no recorded analysis of the upstream terms, and the Hugging Face mirror `PRAIG/fp-grandstaff` tags itself `mit` while simultaneously declaring a non-commercial-only agreement, so the upstream layers are internally inconsistent as well as mutually inconsistent.

  Clarification would still be worth having, from the GrandStaff maintainers (`arios@dlsi.ua.es`, `info-multiscore@dlsi.ua.es`) on what license governs their rendered images, and from Craig Stuart Sapp (`craig@ccrma.stanford.edu`) on the four editions that carry no LICENSE. Note also that Sapp is **not** an author of the GrandStaff paper and is credited nowhere in it beyond a tool citation, so there is no existing relationship to rely on. The released model weights were trained on GrandStaff; training is a separate question from redistribution, but the weights' licensing inherits the unresolved layer as a result.
- **MusicNet and MusicNetEM stack two licenses over the same audio.** MusicNet audio is CC BY 4.0; the MusicNetEM alignment labels are CC BY-NC-SA 4.0. Anything containing EM labels is governed by the more restrictive of the two. Our `musicnet/` layout interleaves both, so they must be separated before either is redistributed under its own terms.
- **MusicNetEM is access-restricted at its source** despite a license that permits non-commercial redistribution. Its Zenodo record requires a personal access token, and its README directs commercial users to contact the author. Obtain it from the author rather than from a mirror.
- **SLakh inherits an unresolved authorship question.** Both SLakh2100 and the Lakh MIDI Dataset it derives from are CC BY 4.0, but Lakh's maintainer explicitly disclaims being able to attribute the underlying MIDI files to their authors. CC licensing cannot cure third-party rights in those arrangements. This exposure is shared by all Lakh derivatives.
- **ASAP data is interleaved into our MAESTRO layout.** `maestro/` contains ASAP's `lmx/` and `asap_note_events/` alongside MAESTRO-derived tokens. Both are CC BY-NC-SA 4.0, so nothing conflicts, but a directory named for one corpus holds two.

### License conflicts in the released artifacts

The Hugging Face releases split into per-corpus token repositories, one codec repository per tokenizer, and one repository for the translation weights. Because Hugging Face assigns exactly **one license tag per repository**, the conflicts above surface as concrete packaging problems.

**Within a single token repository.** Two repositories cannot be tagged truthfully as their contents stand:

- The MusicNet token repository would hold MusicNet-derived audio tokens (CC BY 4.0) beside MusicNetEM note events (CC BY-NC-SA 4.0). No single tag covers both. Either the EM note events move to a non-commercial repository, or the whole repository is tagged CC BY-NC-SA 4.0 and the CC BY 4.0 portion is published under terms stricter than it requires.
- The GrandStaff token repository has no valid tag at all, for the three-layer reason above. It is withheld rather than published under a license that cannot be substantiated.

The MAESTRO token repository holds ASAP-derived notation alongside MAESTRO-derived audio tokens. Both are CC BY-NC-SA 4.0, so the tag is accurate, but the repository name states only one of the two corpora it contains.

**Between token repositories.** The two share-alike families in this collection are mutually incompatible. OLiMPiC tokens are **CC BY-SA 4.0** — share-alike, commercial use permitted. MAESTRO, ASAP and MusicNetEM tokens are **CC BY-NC-SA 4.0** — share-alike, commercial use forbidden. A share-alike license requires derivatives to carry the same license, and these two cannot both be satisfied by one work. Combining tokens across those repositories into a single derivative therefore has no compliant licensing outcome.

**In the trained artifacts.** This matters because the models are trained on exactly that combination. The translation weights and both image codecs are derived from a mixture that includes CC BY-SA 4.0, CC BY-NC-SA 4.0 and CC BY 4.0 material, plus YTSV, whose metadata is CC BY-NC-SA 4.0. Consequently:

- The most restrictive verified terms in the chain are **non-commercial and share-alike**. Any redistribution of the weights, or of audio and notation generated with them, should be treated as bound by both.
- The `-ytsv` suffix on the codec repositories records that they were trained on YTSV-derived data. That provenance alone puts a CC BY-NC-SA 4.0 layer in the chain, independent of the other corpora.
- Publishing a derivative of share-alike material under a license without a share-alike clause does not discharge the share-alike obligation. If you relicense any of these artifacts, check that the chosen license is compatible with every upstream layer, not only with the non-commercial ones.

### A caution on licenses

The license positions above were established from each publisher's own statement and are recorded in good faith, but they are **not legal advice and may be out of date**. Licenses get revised, corpora get relicensed, and several of the statements above are unstated or ambiguous at the source.

Tokenized representations are plausibly derivative works of the corpora they were computed from. If you publish tokens, trained weights, or generated audio derived from this pipeline, the obligations of the underlying corpora — attribution, share-alike, non-commercial restrictions — travel with your output. Several corpora here are non-commercial, so **a model trained on the full mixture cannot be assumed to be free of non-commercial restrictions**.

Verify the current terms with each publisher before you redistribute anything or use it commercially, and satisfy yourself that your intended use is permitted. Where a corpus states no license at all, absence of a statement is not permission.

This work was supported in part by the Ministry of Education of the Republic of Korea, in part by the National Research Foundation of Korea under Grant NRF-2024S1A5C3A03046168, and in part by the IITP of Korea through the Graduate School of Metaverse Convergence Program under Grant RS-2022-00156318.
