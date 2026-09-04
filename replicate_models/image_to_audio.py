"""Replicate model: score image -> piano audio (direct image-to-audio).

Output files (order may vary, so every image carries its label):
  u-must.wav       generated audio
  meta.json        {"duration_sec", "n_systems", "notes"}
  systems.png      all played system crops stacked in playback order
  system_NN.png    each played system crop
"""
from __future__ import annotations

from typing import List

from cog import BasePredictor, Input
from cog import Path

from replicate_models.common import audio_notes, label_image, load_engine, out_dir, select_systems, stack_images, systems_from_image, write_json, write_png, write_wav


class Predictor(BasePredictor):
  def setup(self):
    self.engine = load_engine()

  def predict(
    self,
    image: Path = Input(description="Piano score image: a full page or a single system (PNG/JPG)."),
    system: int = Input(default=1, ge=0, description="1-based index of the system to play; 0 = all detected systems (one pass for up to two systems, Contin-U sliding window beyond that)."),
    seed: int = Input(default=0, description="Random seed; different seeds give different performances."),
  ) -> List[Path]:
    systems = select_systems(systems_from_image(self.engine, str(image)), system)
    res = self.engine.image_to_audio(systems, seed=seed)
    d = out_dir()
    mode = "single window" if len(systems) <= 2 else "Contin-U sliding window"
    files: List[Path] = [Path(write_wav(res, d / "u-must.wav"))]
    files.append(Path(write_json({"duration_sec": round(res.duration, 2), "n_systems": len(systems),
                                  "notes": audio_notes(res, f"{len(systems)} system(s), {mode}.")}, d / "meta.json")))
    n = len(systems)
    labels = [f"System {i + 1} / {n} ({s.label})" for i, s in enumerate(systems)]
    files.append(Path(write_png(stack_images([(s.image, l) for s, l in zip(systems, labels)], width=1600), d / "systems.png")))
    files += [Path(write_png(label_image(s.image, l), d / f"system_{i + 1:02d}.png")) for i, (s, l) in enumerate(zip(systems, labels))]
    return files
