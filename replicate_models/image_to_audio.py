"""Replicate model: score image -> piano audio (named outputs; build with cog 0.16.x, see push.sh)."""

from typing import List, Optional

from cog import BaseModel, BasePredictor, Input
from cog import Path

from replicate_models.common import audio_notes, label_image, load_engine, out_dir, select_systems, stack_images, systems_from_image, write_png, write_wav


class Output(BaseModel):
  audio: Path
  duration_sec: float
  n_systems: int
  systems_image: Optional[Path]
  systems: List[Path]
  notes: str


class Predictor(BasePredictor):
  def setup(self):
    self.engine = load_engine()

  def predict(
    self,
    image: Path = Input(description="Piano score image: a full page or a single system (PNG/JPG)."),
    system: int = Input(default=1, ge=0, description="1-based index of the system to play; 0 = all detected systems (one pass for up to two systems, Contin-U sliding window beyond that)."),
    seed: int = Input(default=0, description="Random seed; different seeds give different performances."),
  ) -> Output:
    systems = select_systems(systems_from_image(self.engine, str(image)), system)
    res = self.engine.image_to_audio(systems, seed=seed)
    d = out_dir()
    n = len(systems)
    labels = [f"System {i + 1} / {n} ({s.label})" for i, s in enumerate(systems)]
    mode = "single window" if n <= 2 else "Contin-U sliding window"
    return Output(
      audio=Path(write_wav(res, d / "u-must.wav")),
      duration_sec=round(res.duration, 2),
      n_systems=n,
      systems_image=Path(write_png(stack_images([(s.image, l) for s, l in zip(systems, labels)], width=1600), d / "systems.png")),
      systems=[Path(write_png(label_image(s.image, l), d / f"system_{i + 1:02d}.png")) for i, (s, l) in enumerate(zip(systems, labels))],
      notes=audio_notes(res, f"{n} system(s), {mode}."),
    )
