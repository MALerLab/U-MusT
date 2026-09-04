# Contin-U · Full PDF piano score to one continuous performance

**Contin-U** turns the U-MusT image-to-audio model, trained on windows of 1–3 systems, into a full-score
renderer without retraining. The PDF is rasterized, every musical system is detected in reading order, and the
model slides over consecutive system pairs (*j*, *j*+1): the decoder is primed with the audio already generated
for system *j*, the cross-attention of a late decoder layer onto the tokens after the `[SEP]` separator marks
where system *j*'s audio ends, the stream is spliced there, and the tokens that follow prime the next window.
The result is a single continuous performance, decoded once from the joined audio tokens.

Contin-U was the MALerLab entry to **RenCon 2025**, the expressive piano performance rendering contest revived
as a MIREX task at ISMIR 2025 (Daejeon). Paper: *Contin-U: Full-Score to Performance Audio with Cross-Attentive
System-Continuation Inference* (Jung, Kim, Lee, Cho, Soh, Bukey, Donahue, Jeong) —
https://futuremirex.com/portal/wp-content/uploads/2025/rencon/Contin-U.pdf

## Inputs

| name | description |
|---|---|
| `score` | PDF of a piano score (engraved or scanned). |
| `first_page`, `last_page` | Page range (1-based; `last_page = 0` = to the end). |
| `dpi` | Rasterization resolution, 150–300. |
| `max_systems` | Stop after this many systems; `0` = all. |
| `attention_threshold` | Cross-attention mass on the second system that marks the boundary (default 0.5). |
| `seed` | Random seed. |

## Outputs

`audio` (WAV, the whole performance), `systems` (system crops in playback order), `n_pages`, `n_systems`,
`duration_sec`, `notes`.

## Notes

- One window (two systems) takes about 30 s of GPU time on an L40S; a 6-system page is 5 windows.
- Piano only; the model expects grand-staff systems. Title pages without music are skipped automatically.
- Research use under CC BY-NC-SA 4.0.

Code: https://github.com/MALerLab/U-MusT · U-MusT paper: https://doi.org/10.1109/TASLPRO.2025.3648794
