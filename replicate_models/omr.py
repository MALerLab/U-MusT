"""Replicate model: optical music recognition (score image -> LMX -> MusicXML).

Output files (order may vary, so every image carries its label):
  transcription.musicxml   (when the systems could be joined into one score)
  transcription.lmx        Linearized MusicXML text, one block per system
  meta.json                {"n_systems", "lmx_tokens", "error"}
  comparison.png           every input system above its engraved transcription, in order
  system_NN.png            each system's transcription engraved with Verovio
  page_NN.png              the joined transcription engraved as pages
"""
from __future__ import annotations

from typing import List

import numpy as np
from cog import BasePredictor, Input
from cog import Path

from replicate_models.common import label_image, load_engine, out_dir, select_systems, stack_images, systems_from_image, write_json, write_png, write_text
from demo.engine import render_lmx_image, render_musicxml_svgs, svg_to_image


class Predictor(BasePredictor):
  def setup(self):
    self.engine = load_engine()

  def predict(
    self,
    image: Path = Input(description="Piano score image: a full page or a single system (PNG/JPG)."),
    system: int = Input(default=0, ge=0, description="1-based index of one detected system to transcribe; 0 = all systems."),
    greedy: bool = Input(default=True, description="Greedy (argmax) decoding; off = sample with `temperature`."),
    temperature: float = Input(default=0.1, ge=0.05, le=1.0, description="Sampling temperature when not greedy."),
    seed: int = Input(default=0, description="Random seed for sampling."),
  ) -> List[Path]:
    systems = select_systems(systems_from_image(self.engine, str(image)), system)
    res = self.engine.omr(systems, greedy=greedy, temperature=temperature, seed=seed)
    d = out_dir()
    files: List[Path] = []

    if res.musicxml:
      files.append(Path(write_text(res.musicxml, d / "transcription.musicxml")))
    lmx_text = "\n\n".join(f"[system {i + 1}]\n{l}" for i, l in enumerate(res.lmx_per_system))
    files.append(Path(write_text(lmx_text, d / "transcription.lmx")))
    files.append(Path(write_json({"n_systems": len(systems), "lmx_tokens": len(res.lmx.split()), "error": res.error}, d / "meta.json")))

    n = len(systems)
    comparison = []
    for i, (crop, lmx) in enumerate(zip(systems, res.lmx_per_system)):
      comparison.append((crop.image, f"System {i + 1} / {n} · input ({crop.label})"))
      img, err = render_lmx_image(lmx, layout="system", width=1600)
      if img is not None:
        comparison.append((img, f"System {i + 1} / {n} · transcription"))
        files.append(Path(write_png(label_image(img, f"System {i + 1} / {n} · transcription"), d / f"system_{i + 1:02d}.png")))
      else:
        comparison.append((np.full((120, 1600, 3), 255, dtype=np.uint8), f"System {i + 1} / {n} · not engraved: {err}"))
    files.append(Path(write_png(stack_images(comparison, width=1600), d / "comparison.png")))
    if res.musicxml:
      pages = render_musicxml_svgs(res.musicxml, layout="page")
      for k, svg in enumerate(pages):
        img = svg_to_image(svg, width=1600)
        if img is not None:
          files.append(Path(write_png(label_image(img, f"Page {k + 1} / {len(pages)} · joined transcription"), d / f"page_{k + 1:02d}.png")))
    return files
