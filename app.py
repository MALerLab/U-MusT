#!/usr/bin/env python3
"""Gradio demo for U-MusT (Image-to-Audio piano model).

Four tabs share one loaded model:

  1. OMR            score image  -> Linearized MusicXML -> MusicXML (+ engraving)
  2. MIDI-to-Audio  performance MIDI -> piano audio (windowed, prefix-conditioned)
  3. Image-to-Audio score image  -> piano audio (systems detected with ls-yolo)
  4. Contin-U       full PDF score -> one continuous performance
                    (two-system sliding window with cross-attention boundaries)

Run locally:   python app.py            (env: UMUST_DEVICE, GRADIO_SERVER_PORT)
On HF Spaces:  see demo/space/README.md
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# optional ZeroGPU support — `spaces` must be imported before torch (or anything
# that initializes CUDA), so this block sits above every other import
# --------------------------------------------------------------------------- #
ZEROGPU = False
try:  # pragma: no cover - only present on Hugging Face Spaces
  import spaces  # type: ignore
  ZEROGPU = bool(os.environ.get("SPACE_ID"))

  def gpu(duration):
    """`duration` is seconds, or a callable of the wrapped function's arguments."""
    return spaces.GPU(duration=duration)
except ImportError:  # local run
  def gpu(duration):  # noqa: ARG001
    def deco(fn):
      return fn
    return deco

import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np

from demo import weights as W
from demo.synth import render_midi_reference
from demo.engine import (
  AudioResult,
  SystemCrop,
  UMusTEngine,
  find_musescore,
  load_image_rgb,
  musicxml_to_pdf,
  pdf_to_page_images,
  piano_roll_image,
  render_lmx_image,
  render_musicxml_svgs,
  svg_to_image,
)

# ZeroGPU compares the *requested* duration with the visitor's remaining daily
# quota (2 min anonymous, 5 min free account, 40 min PRO), so budgets are kept
# tight: one autoregressive window (<= 20 s of audio) costs about
# UMUST_SEC_PER_WINDOW seconds of GPU time. Raise it if tasks get cut off.
SEC_PER_WINDOW = float(os.environ.get("UMUST_SEC_PER_WINDOW", "40"))
SEC_PER_OMR_SYSTEM = float(os.environ.get("UMUST_SEC_PER_OMR_SYSTEM", "8"))


def _omr_budget(systems, choice, *_, **__):
  return int(10 + SEC_PER_OMR_SYSTEM * max(1, len(_select(systems, choice))))


def _midi_windows(window_sec, overlap_sec, max_sec):
  if not max_sec or max_sec <= 0:
    return 8  # unknown file length: assume a few minutes
  return max(1, int(max_sec // max(window_sec - overlap_sec, 1)) + 1)


def _midi_budget(midi_path, window_sec, overlap_sec, max_sec, *_, **__):
  return int(min(10 + SEC_PER_WINDOW * _midi_windows(window_sec, overlap_sec, max_sec), 3600))


def _i2a_budget(systems, choice, *_, **__):
  return int(10 + SEC_PER_WINDOW * max(1, len(_select(systems, choice)) - 1))


def _contin_u_budget(systems, *_, **__):
  return int(min(10 + SEC_PER_WINDOW * max(1, len(systems) - 1), 3600))

EXAMPLES_DIR = W.REPO_ROOT / "demo" / "examples"
OUT_DIR = Path(tempfile.mkdtemp(prefix="umust_demo_"))
DEVICE = os.environ.get("UMUST_DEVICE")  # None -> cuda if available

print("Loading U-MusT engine ...")
ENGINE = UMusTEngine(device=DEVICE)
MSCORE = find_musescore()
print(f"Loaded {ENGINE.checkpoint_name} on {ENGINE.device}; MuseScore: {MSCORE or 'not found'}")

ALL_SYSTEMS = "All systems"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

DISPLAY_W = 1600      # width of the OMR comparison strips and page renders
OVERVIEW_W = 1200     # width of the detected-system lists
MAX_STRIPS = 48       # cap for the detected-system lists (whole-document PDFs)

_CAPTION_STYLE = "font-size:var(--block-label-text-size,14px);color:var(--body-text-color);opacity:.8;margin-bottom:3px"
_BORDER = "1px solid var(--border-color-primary,#8884)"


def _img_data_uri(img: np.ndarray, max_width: int = DISPLAY_W) -> str:
  """Encode a score image as a compact PNG data URI, downscaled for display.

  Score images are line art, so they are stored as a 16-level grayscale
  palette, which is several times smaller than RGB and faster to encode."""
  import base64
  import io
  import PIL.Image
  pil = PIL.Image.fromarray(img)
  if pil.mode != "L":
    pil = pil.convert("L")
  if pil.width > max_width:
    pil = pil.resize((max_width, max(1, round(pil.height * max_width / pil.width))), PIL.Image.LANCZOS)
  pil = pil.quantize(16, dither=PIL.Image.Dither.NONE)
  buf = io.BytesIO()
  pil.save(buf, format="PNG", compress_level=6)
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _strips_html(items, max_height: Optional[int] = None, empty: str = "", max_width: int = DISPLAY_W) -> str:
  """Full-width images stacked vertically at their natural aspect ratio.

  `items` is a list of (image, caption) or (image, caption, is_group_start).
  Wide system crops render as readable strips instead of gallery thumbnails."""
  from html import escape
  if not items:
    return f'<div style="color:var(--body-text-color);opacity:.7;padding:8px">{escape(empty)}</div>' if empty else ""
  figs = []
  for item in items:
    img, caption = item[0], escape(str(item[1]))
    group_start = len(item) > 2 and item[2]
    if img is None:
      figs.append(f'<div style="color:var(--error-text-color,#b55);padding:6px 0">{caption}</div>')
      continue
    border_top = f"border-top:{_BORDER};padding-top:8px;margin-top:10px;" if group_start else ""
    figs.append(
      f'<figure style="margin:0 0 8px 0;{border_top}">'
      f'<figcaption style="{_CAPTION_STYLE}">{caption}</figcaption>'
      f'<img src="{_img_data_uri(img, max_width)}" alt="{caption}" style="width:100%;height:auto;display:block;'
      f'background:#fff;border:{_BORDER};border-radius:4px"/></figure>'
    )
  style = f"max-height:{max_height}px;overflow:auto;" if max_height else ""
  return f'<div style="{style}padding-right:4px">{"".join(figs)}</div>'


def _systems_html(systems: List[SystemCrop], max_height: Optional[int] = 360) -> str:
  items = [(s.image, s.label) for s in systems[:MAX_STRIPS]]
  if len(systems) > MAX_STRIPS:
    items.append((None, f"… {len(systems) - MAX_STRIPS} more systems not shown (all of them are used for generation)"))
  return _strips_html(items, max_height=max_height, max_width=OVERVIEW_W)


PLACEHOLDER_DETECT = _strips_html([], empty="Upload a score image to detect its systems.")
PLACEHOLDER_DOC = _strips_html([], empty="Upload a PDF score and detect its systems.")
PLACEHOLDER_PAIRS = _strips_html([], empty="Transcribe to see each input system above its engraved transcription.")
PLACEHOLDER_PAGES = _strips_html([], empty="The joined transcription is engraved here as pages.")
PLACEHOLDER_PREP = _strips_html([], empty="The normalized system image the model reads appears here.")


def _choices(systems: List[SystemCrop]):
  return [ALL_SYSTEMS] + [s.label for s in systems]


def _select(systems: List[SystemCrop], choice: Optional[str]) -> List[SystemCrop]:
  if not systems:
    return []
  if not choice or choice == ALL_SYSTEMS:
    return list(systems)
  for s in systems:
    if s.label == choice:
      return [s]
  return list(systems)


def _wav_path(audio: np.ndarray, sr: int, stem: str) -> str:
  """Write 16-bit PCM WAV under OUT_DIR with a descriptive, unique name."""
  import re
  import soundfile as sf
  stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "audio"
  path = OUT_DIR / f"{stem}_{abs(hash(audio.tobytes())) % 10**8:08d}.wav"
  sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
  return str(path)


def _audio_out(result: AudioResult, stem: str = "u-must") -> str:
  """Path to a WAV file: shown in the player and offered as a download."""
  return _wav_path(result.audio, result.sample_rate, stem)


def _status(result: AudioResult, extra: str = "") -> str:
  lines = [f"**{result.duration:.1f} s** of audio from **{result.n_tokens}** audio-token steps."]
  if extra:
    lines.append(extra)
  lines += [f"- {n}" for n in result.notes]
  return "\n\n".join(lines)


def _progress_adapter(progress: gr.Progress):
  def fn(fraction: float, desc: str = ""):
    progress(min(max(fraction, 0.0), 1.0), desc=desc)
  return fn


def _error_md(e: Exception) -> str:
  traceback.print_exc()
  return f"⚠️ **{type(e).__name__}**: {e}"


def detect_from_image(image_path: Optional[str]):
  """Shared by the OMR and Image-to-Audio tabs."""
  if not image_path:
    return [], PLACEHOLDER_DETECT, gr.update(choices=[ALL_SYSTEMS], value=ALL_SYSTEMS), "Upload a score image."
  try:
    page = load_image_rgb(image_path)
    systems = ENGINE.systems_from_images([page])
    detected = any(s.conf > 0 for s in systems)
    msg = (f"Detected **{len(systems)}** system(s)." if detected
           else "No system detected by YOLO; treating the whole image as one system.")
    default = systems[0].label if (ZEROGPU and systems) else ALL_SYSTEMS
    return systems, _systems_html(systems), gr.update(choices=_choices(systems), value=default), msg
  except Exception as e:  # noqa: BLE001
    return [], PLACEHOLDER_DETECT, gr.update(choices=[ALL_SYSTEMS], value=ALL_SYSTEMS), _error_md(e)


# --------------------------------------------------------------------------- #
# tab 1: OMR
# --------------------------------------------------------------------------- #

@gpu(duration=_omr_budget)
def run_omr(systems: List[SystemCrop], choice: str, greedy: bool, temperature: float, seed: int,
            progress=gr.Progress()):
  chosen = _select(systems, choice)
  if not chosen:
    return PLACEHOLDER_PAIRS, PLACEHOLDER_PAGES, None, "", "Upload an image first."
  try:
    res = ENGINE.omr(chosen, greedy=greedy, temperature=temperature, seed=int(seed), progress=_progress_adapter(progress))
  except Exception as e:  # noqa: BLE001
    return PLACEHOLDER_PAIRS, PLACEHOLDER_PAGES, None, "", _error_md(e)

  status = [f"Transcribed **{len(chosen)}** system(s), **{len(res.lmx.split())}** LMX tokens."]
  # input crop directly above the engraving of its own transcription, one block per system
  progress(0.9, desc="engraving")
  pairs = []
  for i, (crop, lmx) in enumerate(zip(chosen, res.lmx_per_system)):
    pairs.append((crop.image, f"System {i + 1} · input ({crop.label})", True))
    img, err = render_lmx_image(lmx, layout="system", width=DISPLAY_W)
    if img is not None:
      pairs.append((img, f"System {i + 1} · transcription (Verovio)"))
    else:
      pairs.append((None, f"System {i + 1} · transcription could not be engraved: {err}"))
      status.append(f"System {i + 1} could not be engraved: `{err}`.")

  files, pages = [], []
  if res.musicxml:
    tag = f"omr_{abs(hash(res.lmx)) % 10**8}"
    xml_path = OUT_DIR / f"{tag}.musicxml"
    xml_path.write_text(res.musicxml, encoding="utf-8")
    files.append(str(xml_path))
    for k, svg in enumerate(render_musicxml_svgs(res.musicxml, layout="page")):
      svg_path = OUT_DIR / f"{tag}_page{k + 1}.svg"
      svg_path.write_text(svg, encoding="utf-8")
      files.append(str(svg_path))
      img = svg_to_image(svg, width=DISPLAY_W)
      if img is not None:
        pages.append((img, f"Page {k + 1}"))
  if res.error:
    status.append(f"Joining the systems into one MusicXML failed: `{res.error}`. The raw LMX is still shown.")
  lmx_text = "\n\n".join(f"[system {i + 1}]\n{l}" for i, l in enumerate(res.lmx_per_system))
  return (_strips_html(pairs), _strips_html(pages, max_height=900, empty="No page rendering available."),
          (files or None), lmx_text, "\n\n".join(status))


# --------------------------------------------------------------------------- #
# tab 2: MIDI -> audio
# --------------------------------------------------------------------------- #

def preview_midi(midi_path: Optional[str]):
  """Piano roll plus a plain soundfont rendering of the input, for comparison."""
  if not midi_path:
    return None, None, ""
  try:
    _, _, duration = ENGINE.load_midi_notes(midi_path)
    reference, how = render_midi_reference(midi_path)
    ref_out = None
    if reference is not None:
      sr, audio = reference
      ref_out = _wav_path(audio, sr, f"{Path(midi_path).stem}_soundfont-reference")
      how = f"Reference rendering: {how} (no expressive model involved)."
    else:
      how = f"Reference rendering unavailable: {how}."
    return piano_roll_image(midi_path), ref_out, f"MIDI duration: **{duration:.1f} s**. {how}"
  except Exception as e:  # noqa: BLE001
    return None, None, _error_md(e)


@gpu(duration=_midi_budget)
def run_midi_to_audio(midi_path: Optional[str], window_sec: float, overlap_sec: float, max_sec: float, seed: int,
                      progress=gr.Progress()):
  if not midi_path:
    return None, None, "Upload a MIDI file first."
  try:
    res = ENGINE.midi_to_audio(midi_path, window_sec=window_sec, overlap_sec=overlap_sec,
                               max_duration_sec=max_sec if max_sec > 0 else None, seed=int(seed),
                               progress=_progress_adapter(progress))
    wav = _audio_out(res, f"{Path(midi_path).stem}_u-must_seed{int(seed)}")
    return wav, wav, _status(res)
  except Exception as e:  # noqa: BLE001
    return None, None, _error_md(e)


# --------------------------------------------------------------------------- #
# tab 3: image -> audio
# --------------------------------------------------------------------------- #

@gpu(duration=_i2a_budget)
def run_image_to_audio(systems: List[SystemCrop], choice: str, seed: int, progress=gr.Progress()):
  chosen = _select(systems, choice)
  if not chosen:
    return None, None, PLACEHOLDER_PREP, "Upload an image first."
  try:
    previews = [(ENGINE.preprocess_system(s.image)[1], s.label) for s in chosen]
    res = ENGINE.image_to_audio(chosen, seed=int(seed), progress=_progress_adapter(progress))
    mode = ("single window" if len(chosen) <= 2 else "Contin-U sliding window")
    preview_html = _strips_html(previews, max_height=320, max_width=OVERVIEW_W)
    which = "all-systems" if len(chosen) > 1 else f"p{chosen[0].page + 1}s{chosen[0].index + 1}"
    wav = _audio_out(res, f"image-to-audio_{which}_seed{int(seed)}")
    return wav, wav, preview_html, _status(res, f"{len(chosen)} system(s), {mode}.")
  except Exception as e:  # noqa: BLE001
    return None, None, PLACEHOLDER_PREP, _error_md(e)


# --------------------------------------------------------------------------- #
# tab 4: Contin-U (PDF -> full performance)
# --------------------------------------------------------------------------- #

def detect_from_document(doc_path: Optional[str], first_page: int, last_page: int, dpi: int):
  if not doc_path:
    return [], PLACEHOLDER_DOC, "Upload a PDF score."
  try:
    path = Path(doc_path)
    if path.suffix.lower() in {".mxl", ".musicxml", ".xml"}:
      if MSCORE is None:
        return [], PLACEHOLDER_DOC, "MusicXML input needs MuseScore 3.6.2 (not available here). Please upload a PDF."
      pdf = musicxml_to_pdf(str(path), OUT_DIR / f"{path.stem}.pdf", MSCORE)
      path = pdf
    last = int(last_page) if last_page and last_page > 0 else None
    pages = pdf_to_page_images(str(path), dpi=int(dpi), first_page=int(first_page), last_page=last)
    systems = ENGINE.systems_from_images(pages, fallback_whole_image=False)
    if not systems:
      return [], PLACEHOLDER_DOC, "No musical system detected on the selected pages."
    est = sum(1 for _ in systems)
    msg = (f"**{len(pages)}** page(s), **{len(systems)}** systems detected. "
           f"Generation runs {max(est - 1, 1)} two-system window(s).")
    return systems, _systems_html(systems, max_height=480), msg
  except Exception as e:  # noqa: BLE001
    return [], PLACEHOLDER_DOC, _error_md(e)


@gpu(duration=_contin_u_budget)
def run_contin_u(systems: List[SystemCrop], doc_path: Optional[str], first_page: int, last_page: int, dpi: int,
                 seed: int, attn_thr: float, progress=gr.Progress()):
  if not systems:
    systems, gallery, msg = detect_from_document(doc_path, first_page, last_page, dpi)
    if not systems:
      return None, None, gallery, msg, systems
  else:
    gallery = gr.skip()  # the list is already on screen; do not re-encode it inside the GPU budget
  try:
    res = ENGINE.contin_u(systems, seed=int(seed), attn_threshold=attn_thr, progress=_progress_adapter(progress))
    stem = f"contin-u_{Path(doc_path).stem if doc_path else 'score'}_seed{int(seed)}"
    wav = _audio_out(res, stem)
    return wav, wav, gallery, _status(res, f"{len(systems)} systems stitched with Contin-U."), systems
  except Exception as e:  # noqa: BLE001
    return None, None, gallery, _error_md(e), systems


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

HEADER = f"""
# U-MusT · Score Images ⇄ Symbolic Music ⇄ Performance Audio

Interactive demo of the **Image-to-Audio (piano)** model from
*U-MusT: A Unified Framework for Cross-Modal Translation of Score Images, Symbolic Music, and Performance Audio*
(Jung, Kim et al., IEEE TASLP 2026) and the **Contin-U** full-score inference method.
One encoder–decoder Transformer performs every task below; only the target modality changes.

Score images are cropped into musical systems with the fine-tuned **ls-yolo** system detector and rescaled with the
**staff-height** detector before RQ-VAE tokenization. Audio is decoded with the retrained 44.1 kHz DAC codec.

<sub>Checkpoint `{ENGINE.checkpoint_name}` · device `{ENGINE.device}` · outputs are for research use (CC BY-NC-SA 4.0 weights).</sub>
""" + ("""
> **Running on ZeroGPU.** Each generation window (≤ 20 s of audio) takes roughly half a minute of GPU time, and the
> daily GPU quota is about 2 min for anonymous visitors, 5 min for free accounts and 40 min for PRO. Log in to
> Hugging Face for longer pieces, or start with a single system / a short MIDI excerpt.
""" if ZEROGPU else "")

with gr.Blocks(title="U-MusT demo", theme=gr.themes.Soft()) as demo:
  gr.Markdown(HEADER)

  # ------------------------------------------------------------------ OMR --- #
  with gr.Tab("1 · OMR (Image → MusicXML)"):
    gr.Markdown("Upload a **piano score image** (a whole page or a single system). Each detected system is transcribed "
                "to Linearized MusicXML (LMX), engraved again with Verovio next to its input crop, and all systems are "
                "joined into one downloadable MusicXML file.")
    omr_state = gr.State([])
    with gr.Row():
      with gr.Column(scale=1):
        omr_image = gr.Image(type="filepath", label="Score image", sources=["upload", "clipboard"])
        omr_choice = gr.Dropdown(choices=[ALL_SYSTEMS], value=ALL_SYSTEMS, label="Systems to transcribe")
        with gr.Accordion("Decoding options", open=False):
          omr_greedy = gr.Checkbox(value=True, label="Greedy decoding (argmax)")
          omr_temp = gr.Slider(0.05, 1.0, value=0.1, step=0.05, label="Sampling temperature (when not greedy)")
          omr_seed = gr.Number(value=0, precision=0, label="Seed")
        omr_btn = gr.Button("Transcribe", variant="primary")
        omr_status = gr.Markdown()
      with gr.Column(scale=2):
        with gr.Accordion("Input system vs. engraved transcription (one block per system)", open=True):
          omr_pairs = gr.HTML(PLACEHOLDER_PAIRS)
        with gr.Accordion("Full transcription as pages (Verovio)", open=False):
          omr_pages = gr.HTML(PLACEHOLDER_PAGES)
        with gr.Accordion("Detected systems", open=True):
          omr_gallery = gr.HTML(PLACEHOLDER_DETECT)
        omr_file = gr.File(label="Downloads: MusicXML + SVG pages", file_count="multiple")
        omr_lmx = gr.Textbox(label="LMX tokens", lines=8, show_copy_button=True)
    gr.Examples(examples=[[str(EXAMPLES_DIR / "bach_bwv846_prelude_page1.png")]], inputs=[omr_image],
                label="Example (public-domain engraving from the Mutopia Project)")
    omr_image.change(detect_from_image, [omr_image], [omr_state, omr_gallery, omr_choice, omr_status])
    omr_btn.click(run_omr, [omr_state, omr_choice, omr_greedy, omr_temp, omr_seed],
                  [omr_pairs, omr_pages, omr_file, omr_lmx, omr_status])

  # --------------------------------------------------------- MIDI -> audio --- #
  with gr.Tab("2 · MIDI → Audio"):
    gr.Markdown("Upload a **piano MIDI** file. The model was trained on ≤20 s segments, so the piece is rendered in "
                "overlapping windows: each window's audio is cut to the duration of its MIDI content, the tokens for "
                "the overlap prime the next window as a decoder prefix, and the joined stream is decoded once.")
    with gr.Row():
      with gr.Column(scale=1):
        midi_file = gr.File(label="MIDI file", file_types=[".mid", ".midi"], type="filepath")
        midi_window = gr.Slider(8, 20, value=18, step=0.5, label="Window length (s) — the model was trained on 19–20 s slices")
        midi_overlap = gr.Slider(0, 6, value=2, step=0.5, label="Overlap / conditioning length (s)")
        midi_max = gr.Slider(0, 300, value=20 if ZEROGPU else 60, step=10, label="Max duration to render (s, 0 = whole file)")
        midi_seed = gr.Number(value=0, precision=0, label="Seed")
        midi_btn = gr.Button("Synthesize", variant="primary")
        midi_status = gr.Markdown()
      with gr.Column(scale=2):
        midi_audio = gr.Audio(label="Generated audio (U-MusT)", type="filepath", show_download_button=True)
        midi_dl = gr.File(label="Download WAV")
        midi_ref = gr.Audio(label="Input MIDI rendered with a GM soundfont (FluidSynth reference)", type="filepath",
                            show_download_button=True)
        midi_roll = gr.Image(label="Input piano roll", interactive=False)
    gr.Examples(examples=[[str(EXAMPLES_DIR / "bach_bwv846_prelude.mid")]], inputs=[midi_file], label="Example")
    midi_file.change(preview_midi, [midi_file], [midi_roll, midi_ref, midi_status])
    midi_btn.click(run_midi_to_audio, [midi_file, midi_window, midi_overlap, midi_max, midi_seed],
                   [midi_audio, midi_dl, midi_status])

  # -------------------------------------------------------- image -> audio --- #
  with gr.Tab("3 · Image → Audio"):
    gr.Markdown("Direct score-image-to-audio generation. Pick one system, or all detected systems: one or two systems "
                "are generated in a single pass, more are stitched with the Contin-U sliding window.")
    i2a_state = gr.State([])
    with gr.Row():
      with gr.Column(scale=1):
        i2a_image = gr.Image(type="filepath", label="Score image", sources=["upload", "clipboard"])
        i2a_choice = gr.Dropdown(choices=[ALL_SYSTEMS], value=ALL_SYSTEMS, label="Systems to play")
        i2a_seed = gr.Number(value=0, precision=0, label="Seed")
        i2a_btn = gr.Button("Generate audio", variant="primary")
        i2a_status = gr.Markdown()
      with gr.Column(scale=2):
        i2a_audio = gr.Audio(label="Generated audio", type="filepath", show_download_button=True)
        i2a_dl = gr.File(label="Download WAV")
        with gr.Accordion("Detected systems", open=True):
          i2a_gallery = gr.HTML(PLACEHOLDER_DETECT)
        with gr.Accordion("Model input (staff-height normalized, binarized)", open=False):
          i2a_prep = gr.HTML(PLACEHOLDER_PREP)
    gr.Examples(examples=[[str(EXAMPLES_DIR / "bach_bwv846_prelude_page1.png")]], inputs=[i2a_image], label="Example")
    i2a_image.change(detect_from_image, [i2a_image], [i2a_state, i2a_gallery, i2a_choice, i2a_status])
    i2a_btn.click(run_image_to_audio, [i2a_state, i2a_choice, i2a_seed], [i2a_audio, i2a_dl, i2a_prep, i2a_status])

  # --------------------------------------------------------------- Contin-U --- #
  with gr.Tab("4 · Contin-U (PDF → full performance)"):
    gr.Markdown(
      "Upload a **PDF piano score**" + (" (or MusicXML, engraved with MuseScore 3.6.2)" if MSCORE else "") +
      ". Pages are rasterized, systems are detected and ordered, and the model slides over consecutive system pairs. "
      "Cross-attention to the `[SEP]` token marks where the audio of system *j* ends; that boundary splices the "
      "windows and the following tokens prime the next window, giving one continuous performance without retraining.\n\n"
      "**Contin-U** was the MALerLab entry to **RenCon 2025**, the performance rendering contest held with "
      "MIREX 2025: the contest supplies MusicXML scores, which are engraved to page images and fed to this unchanged "
      "image-to-audio model, so phrasing, rubato and dynamics come entirely from the model. "
      "Paper: [Contin-U: Full-Score to Performance Audio with Cross-Attentive System-Continuation Inference]"
      "(https://futuremirex.com/portal/wp-content/uploads/2025/rencon/Contin-U.pdf) (Jung, Kim, Lee, Cho, Soh, Bukey, "
      "Donahue, Jeong)."
    )
    cu_state = gr.State([])
    with gr.Row():
      with gr.Column(scale=1):
        cu_doc = gr.File(label="Score (PDF" + (", MusicXML" if MSCORE else "") + ")",
                         file_types=[".pdf"] + ([".mxl", ".musicxml", ".xml"] if MSCORE else []), type="filepath")
        with gr.Row():
          cu_first = gr.Number(value=1, precision=0, label="First page")
          cu_last = gr.Number(value=1 if ZEROGPU else 0, precision=0, label="Last page (0 = end)")
        cu_dpi = gr.Slider(150, 300, value=300, step=50, label="Rasterization DPI")
        with gr.Accordion("Generation options", open=False):
          cu_seed = gr.Number(value=0, precision=0, label="Seed")
          cu_thr = gr.Slider(0.3, 0.8, value=0.5, step=0.05, label="Attention boundary threshold")
        with gr.Row():
          cu_detect = gr.Button("Detect systems")
          cu_btn = gr.Button("Generate full audio", variant="primary")
        cu_status = gr.Markdown()
      with gr.Column(scale=2):
        cu_audio = gr.Audio(label="Continuous performance", type="filepath", show_download_button=True)
        cu_dl = gr.File(label="Download WAV")
        with gr.Accordion("Systems in playback order", open=True):
          cu_gallery = gr.HTML(PLACEHOLDER_DOC)
    gr.Examples(examples=[[str(EXAMPLES_DIR / "bach_bwv846_prelude.pdf")]], inputs=[cu_doc],
                label="Example (J. S. Bach, Prelude in C major BWV 846 — Mutopia Project, public domain)")
    cu_doc.change(detect_from_document, [cu_doc, cu_first, cu_last, cu_dpi], [cu_state, cu_gallery, cu_status])
    cu_detect.click(detect_from_document, [cu_doc, cu_first, cu_last, cu_dpi], [cu_state, cu_gallery, cu_status])
    cu_btn.click(run_contin_u, [cu_state, cu_doc, cu_first, cu_last, cu_dpi, cu_seed, cu_thr],
                 [cu_audio, cu_dl, cu_gallery, cu_status, cu_state])

  gr.Markdown(
    "Paper: [IEEE TASLP 10.1109/TASLPRO.2025.3648794](https://doi.org/10.1109/TASLPRO.2025.3648794) · "
    "Contin-U: [RenCon 2025 / MIREX paper](https://futuremirex.com/portal/wp-content/uploads/2025/rencon/Contin-U.pdf) · "
    "Code: [MALerLab/U-MusT](https://github.com/MALerLab/U-MusT) · "
    "Weights: [malerlab/u-must](https://huggingface.co/malerlab/u-must) · "
    "Audio examples: [sakem.in/u-must](https://sakem.in/u-must/)"
  )

if __name__ == "__main__":
  demo.queue(default_concurrency_limit=1).launch(
    server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    share=os.environ.get("GRADIO_SHARE", "0") == "1",
  )
