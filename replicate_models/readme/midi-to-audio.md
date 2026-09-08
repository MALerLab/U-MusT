# U-MusT · MIDI to Audio (piano)

Renders a **performance MIDI file** as piano audio with the U-MusT translation model: MT3-style MIDI event tokens
go into the encoder, and the decoder emits DAC audio tokens that are decoded to 44.1 kHz audio. Timing follows
the MIDI, so this is a neural synthesizer rather than a performance model. This is the **MIDI-to-audio
fine-tuned checkpoint** (`run-20250330_182257-cogdba9o`): the multi-task piano model further fine-tuned on
MAESTRO for the MIDI-to-audio task alone, as reported in the paper.

The fine-tuning used 10 s slices, so longer files are rendered in overlapping windows (default window = the
native slice length, `window_sec = 0`): each
window's audio is cut to the duration of its MIDI content, the audio tokens of the overlap prime the next
window as a decoder prefix, and the joined token stream is decoded once. A plain General-MIDI soundfont
rendering of the same file (FluidSynth, FluidR3 GM) is returned alongside for comparison.

## Inputs

| name | description |
|---|---|
| `midi` | Standard MIDI file; piano tracks (every program is mapped to piano). |
| `window_sec` | Window length in seconds; `0` (default) = the checkpoint's native 10 s slices. |
| `overlap_sec` | Overlap primed into the next window, 0–6 s (default 2). |
| `max_duration_sec` | Render only the first N seconds; `0` = whole file. |
| `seed` | Random seed. |
| `reference` | Also return the soundfont rendering (default on). |

## Outputs

`audio` (WAV, the model's rendition), `reference_audio` (WAV, the soundfont rendering, when `reference` is on),
`duration_sec`, `n_tokens`, `notes`.

## Notes

- Piano only. Sustain pedal is applied when converting the MIDI to notes.
- About 15 s of GPU time per 10 s window on an L40S; a 3-minute piece takes roughly 22 windows.
- Research use under CC BY-NC-SA 4.0.

Code: https://github.com/MALerLab/U-MusT · Paper: https://doi.org/10.1109/TASLPRO.2025.3648794
