"""Replicate model: performance MIDI -> piano audio.

Output files, in order:
  u-must.wav       the model's rendition
  reference.wav    the input MIDI played with a General MIDI soundfont (when `reference` is on)
  meta.json        {"duration_sec", "n_tokens", "notes"}
"""
from __future__ import annotations

from typing import List

import numpy as np
from cog import BasePredictor, Input
from cog import Path

from replicate_models.common import audio_notes, load_engine, out_dir, write_json, write_wav
from demo.synth import render_midi_reference


class Predictor(BasePredictor):
  def setup(self):
    self.engine = load_engine()

  def predict(
    self,
    midi: Path = Input(description="Piano MIDI file (.mid)."),
    window_sec: float = Input(default=18.0, ge=8.0, le=20.0, description="Generation window length in seconds (the model was trained on 19-20 s slices)."),
    overlap_sec: float = Input(default=2.0, ge=0.0, le=6.0, description="Overlap between windows; its generated audio primes the next window."),
    max_duration_sec: float = Input(default=0.0, ge=0.0, description="Only render the first N seconds of the MIDI; 0 = whole file."),
    seed: int = Input(default=0, description="Random seed."),
    reference: bool = Input(default=True, description="Also return the input MIDI rendered with a General MIDI soundfont, for comparison."),
  ) -> List[Path]:
    res = self.engine.midi_to_audio(
      str(midi), window_sec=window_sec, overlap_sec=overlap_sec,
      max_duration_sec=max_duration_sec if max_duration_sec > 0 else None, seed=seed,
    )
    d = out_dir()
    files: List[Path] = [Path(write_wav(res, d / "u-must.wav"))]

    how = ""
    if reference:
      ref, how = render_midi_reference(str(midi), max_duration_sec=max_duration_sec if max_duration_sec > 0 else None)
      if ref is not None:
        import soundfile as sf
        sr, wav = ref
        sf.write(str(d / "reference.wav"), np.clip(wav, -1, 1), sr, subtype="PCM_16")
        files.append(Path(d / "reference.wav"))
        how = f"Reference: {how}."
      else:
        how = f"Reference unavailable: {how}."

    files.append(Path(write_json({"duration_sec": round(res.duration, 2), "n_tokens": res.n_tokens,
                                  "notes": audio_notes(res, how)}, d / "meta.json")))
    return files
