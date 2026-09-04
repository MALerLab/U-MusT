# U-MusT · Image to Audio (piano)

Generates piano audio **directly from a score image** — no OMR, MIDI or performance-rendering step in between.
Systems are detected with the ls-yolo detector, rescaled to the training staff height, tokenized with the
RQ-VAE image codec, and the U-MusT encoder–decoder Transformer emits DAC audio tokens that are decoded to
44.1 kHz audio. Phrasing, tempo and dynamics are implicit in the model, learned from 815 hours of paired
score-video/audio data (YTSV-P).

## Inputs

| name | description |
|---|---|
| `image` | PNG/JPG of a piano score: a whole page or a single system. |
| `system` | 1-based index of the system to play (default 1); `0` = all detected systems. One or two systems are generated in a single pass; more are stitched with the Contin-U sliding window (see `malerlab/u-must-contin-u`). |
| `seed` | Random seed; different seeds give different performances. |

## Outputs

`audio` (WAV), `systems` (the system crops that were played, in order), `n_systems`, `duration_sec`, `notes`.

## Notes

- The model was trained on 1–3 systems / up to 20 s of audio per sample; a single system usually yields
  10–20 s of audio. Very dense systems may be truncated at the 20 s token budget.
- Piano only; research use under CC BY-NC-SA 4.0.

Code: https://github.com/MALerLab/U-MusT · Paper: https://doi.org/10.1109/TASLPRO.2025.3648794
