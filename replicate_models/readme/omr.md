# U-MusT · Optical Music Recognition (piano)

Transcribes a **piano score image** into notation. Every musical system on the page is detected with the
fine-tuned ls-yolo detector, rescaled to the training staff height, tokenized with the RQ-VAE image codec and
translated by the U-MusT encoder–decoder Transformer into **Linearized MusicXML (LMX)**. The systems are joined
into one MusicXML file, and each transcription is engraved again with Verovio so it can be compared with the input.

Model: *U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance
Audio* (Jung, Kim et al., IEEE TASLP 2026) — the released Image-to-Audio piano checkpoint, which was trained
jointly on OMR, MIDI-to-audio and image-to-audio (OLiMPiC scanned SER 13.67 %).

## Inputs

| name | description |
|---|---|
| `image` | PNG/JPG of a piano score: a whole page or a single system (grand staff). Scans and engraved pages both work; 200–300 dpi recommended. |
| `system` | 1-based index of one detected system to transcribe; `0` transcribes all systems in reading order. |
| `greedy` | Greedy decoding (default). Turn off to sample with `temperature`. |
| `temperature`, `seed` | Sampling controls when `greedy` is off. |

## Outputs

A list of files (order may vary): `transcription.musicxml` (when the systems could be joined into one score),
`transcription.lmx` (the LMX text, one block per system), `meta.json` (`n_systems`, `lmx_tokens`, `error`),
`system_NN.png` (each system's transcription engraved with Verovio) and `page_NN.png` (the joined transcription
engraved as pages).

## Notes

- Piano (grand staff) only; the model was trained on GrandStaff, OLiMPiC and YTSV-P.
- Output notes are pitch-exact in LMX; on some pages the model restates a treble clef on the lower staff of later
  systems, which affects the engraving but not the pitches.
- Research use under CC BY-NC-SA 4.0 (the weights inherit non-commercial terms from their training corpora).

Code: https://github.com/MALerLab/U-MusT · Paper: https://doi.org/10.1109/TASLPRO.2025.3648794
