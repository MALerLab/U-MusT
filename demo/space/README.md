---
title: U-MusT
emoji: 🎼
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Score image / MIDI → piano audio, OMR, and Contin-U
models:
  - malerlab/u-must
  - malerlab/unirqvae3-ytsv
  - malerlab/unidac4-ytsv
tags:
  - music
  - audio-generation
  - optical-music-recognition
  - midi-to-audio
---

# U-MusT demo

Interactive demo of the **Image-to-Audio (piano)** models from

> **U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance Audio**
> Jongmin Jung\*, Dongmin Kim\*, Sihun Lee, Seola Cho, Hyungjoon Soh, Irmak Bukey, Chris Donahue, Dasaem Jeong
> *IEEE Transactions on Audio, Speech and Language Processing*, vol. 34, 2026 · [doi:10.1109/TASLPRO.2025.3648794](https://doi.org/10.1109/TASLPRO.2025.3648794)

and of **Contin-U**, the two-system sliding-window inference that turns the same model into a full-score renderer.
Contin-U was the MALerLab entry to [RenCon 2025](https://ren-con2025.vercel.app/), the expressive piano performance
rendering contest held as a MIREX task at ISMIR 2025 ([paper](https://futuremirex.com/portal/wp-content/uploads/2025/rencon/Contin-U.pdf)).

| Tab | Input | Output |
|---|---|---|
| OMR | score image (page or system) | Linearized MusicXML → MusicXML, engraved with Verovio |
| MIDI → Audio | performance MIDI | 44.1 kHz piano audio, windowed with audio-prefix conditioning |
| Image → Audio | score image | piano audio generated directly from the image tokens |
| Contin-U | PDF score | one continuous performance across every system and page |

Score images are cropped into systems with the [ls-yolo](https://github.com/MALerLab/ls-yolo) system detector and
rescaled to an 18 px staff height with the staff-height detector before RQ-VAE tokenization.

Three checkpoints from [malerlab/u-must](https://huggingface.co/malerlab/u-must) serve the tabs, as in the paper:
the multi-task piano run (`run-20250225_062905-9n1554as`) for image-to-audio and Contin-U, and its per-task
fine-tuned runs for OMR (`run-20250302_101330-hhpxlltr`) and MIDI-to-audio (`run-20250330_182257-cogdba9o`),
selected with the `UMUST_RUN_OMR` / `UMUST_RUN_MIDI` variables.

## Deploying this Space

The translation weights ([malerlab/u-must](https://huggingface.co/malerlab/u-must)) are **gated**. Add a repository
secret named `HF_TOKEN` whose owner has been granted access (or set the `UMUST_HF_WEIGHTS_REPO` variable to a mirror
with the same layout); the app downloads the piano run, both codecs and the YOLO detectors on first start.

A GPU (~6 GB) is required. The Space is written for **ZeroGPU** — the only GPU tier a free personal account can host
(up to two Spaces): `torch` is pinned to a ZeroGPU-supported release, the GPU-bound handlers are wrapped in
`@spaces.GPU` with per-call durations derived from the number of generation windows, and the model is placed on
`cuda` at import time as ZeroGPU requires. `UMUST_SEC_PER_WINDOW` (default 40) tunes the requested duration per
window. Dedicated hardware (T4 or better) works unchanged.

Code: [MALerLab/U-MusT](https://github.com/MALerLab/U-MusT) · Audio examples: [sakem.in/u-must](https://sakem.in/u-must/)

The example score (J. S. Bach, Prelude in C major BWV 846) is a public-domain engraving from the
[Mutopia Project](https://www.mutopiaproject.org/). Generated audio inherits the CC BY-NC-SA 4.0 terms of the weights.
