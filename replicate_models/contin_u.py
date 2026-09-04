"""Replicate model: Contin-U — a PDF piano score -> one continuous performance.

Output files, in order:
  contin-u.wav     the whole performance
  meta.json        {"duration_sec", "n_pages", "n_systems", "notes"}
  pPP_sSS.png      the system crops in playback order (page, system)
"""
from __future__ import annotations

from typing import List

from cog import BasePredictor, Input
from cog import Path

from replicate_models.common import audio_notes, load_engine, out_dir, pdf_to_page_images, write_json, write_png, write_wav


class Predictor(BasePredictor):
  def setup(self):
    self.engine = load_engine()

  def predict(
    self,
    score: Path = Input(description="Piano score as a PDF (pages are rasterized and every system is detected in reading order)."),
    first_page: int = Input(default=1, ge=1, description="First page to render (1-based)."),
    last_page: int = Input(default=0, ge=0, description="Last page to render; 0 = until the end."),
    dpi: int = Input(default=300, ge=150, le=300, description="Rasterization resolution."),
    seed: int = Input(default=0, description="Random seed."),
    attention_threshold: float = Input(default=0.5, ge=0.3, le=0.8, description="Cross-attention mass on the second system that marks the boundary between two systems' audio."),
    max_systems: int = Input(default=0, ge=0, description="Stop after this many systems; 0 = all."),
  ) -> List[Path]:
    pages = pdf_to_page_images(str(score), dpi=dpi, first_page=first_page, last_page=last_page if last_page > 0 else None)
    systems = self.engine.systems_from_images(pages, fallback_whole_image=False)
    if not systems:
      raise ValueError("no musical system detected on the selected pages")
    if max_systems > 0:
      systems = systems[:max_systems]
    res = self.engine.contin_u(systems, seed=seed, attn_threshold=attention_threshold)
    d = out_dir()
    files: List[Path] = [Path(write_wav(res, d / "contin-u.wav"))]
    files.append(Path(write_json({"duration_sec": round(res.duration, 2), "n_pages": len(pages), "n_systems": len(systems),
                                  "notes": audio_notes(res, f"{len(systems)} systems stitched with Contin-U ({max(len(systems) - 1, 1)} windows).")},
                                 d / "meta.json")))
    files += [Path(write_png(s.image, d / f"p{s.page + 1:02d}_s{s.index + 1:02d}.png")) for s in systems]
    return files
