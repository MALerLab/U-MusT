"""Replicate model: optical music recognition (score image -> LMX -> MusicXML).

Outputs are named fields of a BaseModel; the images carry burnt-in labels
and `comparison` stacks every input system above its transcription in
order, because Replicate does not keep list order. Build with the legacy
cog CLI (0.16.x): the 0.17+ runtime does not upload files nested in a
BaseModel (see replicate_models/push.sh).
"""

from typing import List, Optional

import numpy as np
from cog import BaseModel, BasePredictor, Input
from cog import Path

from replicate_models.common import label_image, load_engine, out_dir, select_systems, stack_images, systems_from_image, write_png, write_text
from demo.engine import render_lmx_image, render_musicxml_svgs, svg_to_image


class Output(BaseModel):
  musicxml: Optional[Path]
  lmx: str
  n_systems: int
  comparison: Optional[Path]
  transcriptions: List[Path]
  pages: List[Path]
  error: Optional[str]


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
  ) -> Output:
    systems = select_systems(systems_from_image(self.engine, str(image)), system)
    res = self.engine.omr(systems, greedy=greedy, temperature=temperature, seed=seed)
    d = out_dir()
    n = len(systems)

    comparison, transcriptions = [], []
    for i, (crop, lmx) in enumerate(zip(systems, res.lmx_per_system)):
      comparison.append((crop.image, f"System {i + 1} / {n} · input ({crop.label})"))
      img, err = render_lmx_image(lmx, layout="system", width=1600)
      if img is not None:
        comparison.append((img, f"System {i + 1} / {n} · transcription"))
        transcriptions.append(Path(write_png(label_image(img, f"System {i + 1} / {n} · transcription"), d / f"system_{i + 1:02d}.png")))
      else:
        comparison.append((np.full((120, 1600, 3), 255, dtype=np.uint8), f"System {i + 1} / {n} · not engraved: {err}"))
    comparison_path = Path(write_png(stack_images(comparison, width=1600), d / "comparison.png"))

    musicxml, pages = None, []
    if res.musicxml:
      musicxml = Path(write_text(res.musicxml, d / "transcription.musicxml"))
      svgs = render_musicxml_svgs(res.musicxml, layout="page")
      for k, svg in enumerate(svgs):
        img = svg_to_image(svg, width=1600)
        if img is not None:
          pages.append(Path(write_png(label_image(img, f"Page {k + 1} / {len(svgs)} · joined transcription"), d / f"page_{k + 1:02d}.png")))

    return Output(
      musicxml=musicxml,
      lmx="\n\n".join(f"[system {i + 1}]\n{l}" for i, l in enumerate(res.lmx_per_system)),
      n_systems=n,
      comparison=comparison_path,
      transcriptions=transcriptions,
      pages=pages,
      error=res.error,
    )
