# U-MusT · MIDI to Audio (piano)

Renders a **performance MIDI file** as piano audio with the U-MusT translation model: MT3-style MIDI event tokens
go into the encoder, and the decoder emits DAC audio tokens that are decoded to 44.1 kHz audio. Timing follows
the MIDI, so this is a neural synthesizer rather than a performance model; the same checkpoint also performs OMR
and direct image-to-audio.

The model was trained on 19–20 s slices (MAESTRO), so longer files are rendered in overlapping windows: each
window's audio is cut to the duration of its MIDI content, the audio tokens of the overlap prime the next
window as a decoder prefix, and the joined token stream is decoded once. A plain General-MIDI soundfont
rendering of the same file (FluidSynth, FluidR3 GM) is returned alongside for comparison.

## Inputs

| name | description |
|---|---|
| `midi` | Standard MIDI file; piano tracks (every program is mapped to piano). |
| `window_sec` | Window length, 8–20 s (default 18). |
| `overlap_sec` | Overlap primed into the next window, 0–6 s (default 2). |
| `max_duration_sec` | Render only the first N seconds; `0` = whole file. |
| `seed` | Random seed. |
| `reference` | Also return the soundfont rendering (default on). |

## Outputs

`audio` (WAV, U-MusT), `reference_audio` (WAV, soundfont), `duration_sec`, `n_tokens`, `notes`.

## Notes

- Piano only. Sustain pedal is applied when converting the MIDI to notes.
- About 30 s of GPU time per 18 s window on an L40S; a 3-minute piece takes roughly 10 windows.
- Research use under CC BY-NC-SA 4.0.

Code: https://github.com/MALerLab/U-MusT · Paper: https://doi.org/10.1109/TASLPRO.2025.3648794
