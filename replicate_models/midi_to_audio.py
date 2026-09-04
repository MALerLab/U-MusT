"""Replicate model: performance MIDI -> piano audio (named outputs; build with cog 0.16.x, see push.sh)."""

from typing import Optional

import numpy as np
from cog import BaseModel, BasePredictor, Input
from cog import Path

from replicate_models.common import audio_notes, load_engine, out_dir, write_wav
from demo.synth import render_midi_reference


class Output(BaseModel):
  audio: Path
  reference_audio: Optional[Path]
  duration_sec: float
  n_tokens: int
  notes: str


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
  ) -> Output:
    res = self.engine.midi_to_audio(
      str(midi), window_sec=window_sec, overlap_sec=overlap_sec,
      max_duration_sec=max_duration_sec if max_duration_sec > 0 else None, seed=seed,
    )
    d = out_dir()
    audio = write_wav(res, d / "u-must.wav")

    ref_path, how = None, ""
    if reference:
      ref, how = render_midi_reference(str(midi), max_duration_sec=max_duration_sec if max_duration_sec > 0 else None)
      if ref is not None:
        import soundfile as sf
        sr, wav = ref
        ref_path = d / "reference.wav"
        sf.write(str(ref_path), np.clip(wav, -1, 1), sr, subtype="PCM_16")
        how = f"Reference: {how}."
      else:
        how = f"Reference unavailable: {how}."

    return Output(
      audio=Path(audio),
      reference_audio=Path(ref_path) if ref_path else None,
      duration_sec=round(res.duration, 2),
      n_tokens=res.n_tokens,
      notes=audio_notes(res, how),
    )
