"""Replicate model: score image -> piano audio (direct image-to-audio)."""
from __future__ import annotations

from typing import List

from cog import BaseModel, BasePredictor, Input
from cog import Path

from replicate_models.common import audio_notes, load_engine, out_dir, select_systems, systems_from_image, write_png, write_wav


class Output(BaseModel):
  audio: Path
  duration_sec: float
  n_systems: int
  systems: List[Path]
  notes: str


class Predictor(BasePredictor):
  def setup(self):
    self.engine = load_engine()

  def predict(
    self,
    image: Path = Input(description="Piano score image: a full page or a single system (PNG/JPG)."),
    system: int = Input(default=1, ge=0, description="1-based index of the system to play; 0 = all detected systems (one pass for up to two systems, Contin-U sliding window beyond that)."),
    seed: int = Input(default=0, description="Random seed."),
  ) -> Output:
    systems = select_systems(systems_from_image(self.engine, str(image)), system)
    res = self.engine.image_to_audio(systems, seed=seed)
    d = out_dir()
    crops = [Path(write_png(s.image, d / f"system_{i + 1:02d}.png")) for i, s in enumerate(systems)]
    mode = "single window" if len(systems) <= 2 else "Contin-U sliding window"
    return Output(
      audio=Path(write_wav(res, d / "u-must.wav")),
      duration_sec=round(res.duration, 2),
      n_systems=len(systems),
      systems=crops,
      notes=audio_notes(res, f"{len(systems)} system(s), {mode}."),
    )
