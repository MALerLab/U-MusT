"""Replicate model: optical music recognition (score image -> LMX -> MusicXML)."""
from __future__ import annotations

from typing import List, Optional

from cog import BaseModel, BasePredictor, Input
from cog import Path

from replicate_models.common import load_engine, out_dir, select_systems, systems_from_image, write_png
from demo.engine import render_lmx_image, render_musicxml_svgs, svg_to_image


class Output(BaseModel):
  musicxml: Optional[Path]
  lmx: str
  n_systems: int
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

    transcriptions = []
    for i, lmx in enumerate(res.lmx_per_system):
      img, _ = render_lmx_image(lmx, layout="system", width=1600)
      if img is not None:
        transcriptions.append(Path(write_png(img, d / f"system_{i + 1:02d}.png")))

    musicxml, pages = None, []
    if res.musicxml:
      xml_path = d / "transcription.musicxml"
      xml_path.write_text(res.musicxml, encoding="utf-8")
      musicxml = Path(xml_path)
      for k, svg in enumerate(render_musicxml_svgs(res.musicxml, layout="page")):
        img = svg_to_image(svg, width=1600)
        if img is not None:
          pages.append(Path(write_png(img, d / f"page_{k + 1:02d}.png")))

    return Output(
      musicxml=musicxml,
      lmx="\n\n".join(f"[system {i + 1}]\n{l}" for i, l in enumerate(res.lmx_per_system)),
      n_systems=len(systems),
      transcriptions=transcriptions,
      pages=pages,
      error=res.error,
    )
