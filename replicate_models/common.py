"""Shared helpers for the Replicate predictors.

Outputs are returned as a flat, ordered list of files (cog's runtime only
uploads top-level `Path` / `List[Path]` outputs; files nested inside a
`BaseModel` are inlined as data URIs, which Replicate does not store), so
every model also writes a `meta.json` with its scalar results.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

# Weights are downloaded at image build time (see the `run` step in the
# cog.*.yaml files) into /weights; fall back to the repository layout when the
# predictor runs outside the image (e.g. `cog predict` with local weights).
_WEIGHTS = Path(os.environ.get("UMUST_WEIGHTS_ROOT", "/weights"))
if _WEIGHTS.exists():
  os.environ.setdefault("UMUST_MODELS_DIR", str(_WEIGHTS / "models"))
  os.environ.setdefault("UMUST_VQ_MODEL_DIR", str(_WEIGHTS / "vq_models"))
  os.environ.setdefault("UMUST_DAC_MODEL_DIR", str(_WEIGHTS / "dac_models"))
  os.environ.setdefault("UMUST_YOLO_DIR", str(_WEIGHTS / "yolo"))
os.environ.setdefault("UMUST_YOLO_DEVICE", "cpu")

from demo.engine import AudioResult, SystemCrop, UMusTEngine, load_image_rgb, pdf_to_page_images  # noqa: E402


def load_engine() -> UMusTEngine:
  return UMusTEngine(device="cuda", download=False)


def out_dir() -> Path:
  return Path(tempfile.mkdtemp(prefix="umust_"))


def write_wav(result: AudioResult, path: Path) -> Path:
  import soundfile as sf
  sf.write(str(path), np.clip(result.audio, -1.0, 1.0), result.sample_rate, subtype="PCM_16")
  return path


def write_png(img: np.ndarray, path: Path) -> Path:
  import PIL.Image
  PIL.Image.fromarray(img).save(str(path))
  return path


def label_image(img: np.ndarray, text: str, scale: float = 1.0) -> np.ndarray:
  """Burn a caption into the top-left corner (Replicate shows images without
  their file names and does not keep the output order)."""
  import PIL.Image
  import PIL.ImageDraw
  import PIL.ImageFont
  pil = PIL.Image.fromarray(img).convert("RGB")
  size = max(18, int(min(pil.width, 1600) / 45 * scale))
  try:
    font = PIL.ImageFont.load_default(size=size)
  except TypeError:  # Pillow < 10.1
    font = PIL.ImageFont.load_default()
  draw = PIL.ImageDraw.Draw(pil)
  x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
  pad = size // 3
  banner = PIL.Image.new("RGB", (pil.width, y1 - y0 + 2 * pad + 4), (255, 255, 255))
  out = PIL.Image.new("RGB", (pil.width, banner.height + pil.height), (255, 255, 255))
  out.paste(banner, (0, 0))
  out.paste(pil, (0, banner.height))
  draw = PIL.ImageDraw.Draw(out)
  draw.rectangle([0, 0, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad], fill=(40, 40, 40))
  draw.text((pad - x0, pad - y0), text, font=font, fill=(255, 255, 255))
  return np.array(out)


def stack_images(items, width: int = 1600, gap: int = 24) -> np.ndarray:
  """Stack (image, caption) pairs vertically at a common width, in order."""
  import PIL.Image
  tiles = []
  for img, caption in items:
    pil = PIL.Image.fromarray(label_image(img, caption)).convert("RGB")
    if pil.width != width:
      pil = pil.resize((width, max(1, round(pil.height * width / pil.width))), PIL.Image.LANCZOS)
    tiles.append(pil)
  height = sum(t.height for t in tiles) + gap * max(0, len(tiles) - 1)
  out = PIL.Image.new("RGB", (width, max(1, height)), (255, 255, 255))
  y = 0
  for t in tiles:
    out.paste(t, (0, y))
    y += t.height + gap
  return np.array(out)


def write_text(text: str, path: Path) -> Path:
  path.write_text(text, encoding="utf-8")
  return path


def write_json(obj: dict, path: Path) -> Path:
  import json
  path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
  return path


def systems_from_image(engine: UMusTEngine, image_path: str) -> List[SystemCrop]:
  page = load_image_rgb(image_path)
  return engine.systems_from_images([page], fallback_whole_image=True)


def select_systems(systems: List[SystemCrop], index: int) -> List[SystemCrop]:
  """`index` is 1-based; 0 means all systems."""
  if index <= 0:
    return list(systems)
  if index > len(systems):
    raise ValueError(f"system {index} requested but only {len(systems)} detected")
  return [systems[index - 1]]


def audio_notes(result: AudioResult, extra: str = "") -> str:
  parts = [f"{result.duration:.1f} s of audio from {result.n_tokens} audio-token steps."]
  if extra:
    parts.append(extra)
  parts += result.notes
  return " ".join(parts)
