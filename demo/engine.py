"""Inference engine behind the U-MusT Gradio demo.

Everything here is independent of Gradio so it can also be driven from a
script or a notebook. One `UMusTEngine` instance holds the I2A translation
model, the RQ-VAE image codec, the DAC audio codec and the two ls-yolo
detectors, and exposes four tasks:

  * `omr(images)`               score image(s)  -> LMX + MusicXML
  * `image_to_audio(images)`    score image(s)  -> performance audio
  * `midi_to_audio(midi_path)`  performance MIDI -> performance audio
  * `contin_u(system_images)`   ordered system crops -> one continuous audio

The Contin-U loop is a port of `infer.py` (two-system sliding window with
cross-attention boundary detection); the MIDI path reuses the same
audio-prefix conditioning to stitch consecutive windows.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision
from omegaconf import OmegaConf, open_dict

from umust.utils import (
  convert_wandb_style_config_to_omega_config,
  get_dac_model,
  get_model,
  get_vq_model,
  load_model_state_dict,
)
from umust.vocab_utils import LMXVocab, RVQVocab, TokenIdxHandler
from umust.midi_utils.tokenizer import NoteEventTokenizer
from umust.midi_utils.midi import midi2note
from umust.midi_utils.note2event import note2note_event, slice_note_events_and_ties
from umust.lmx_utils import delinearize_lmx
from umust.evaluation_utils import LayerPeeper, use_attn_weights

from demo import weights as W

ProgressFn = Callable[[float, str], None]

DAC_TOKENS_PER_SEC = 44100 / 512  # 86.13 token sets per second


def _noop_progress(fraction: float, desc: str = ""):
  return None


# --------------------------------------------------------------------------- #
# config / vocab
# --------------------------------------------------------------------------- #

def load_run_config(run_path: Path) -> OmegaConf:
  config = OmegaConf.load(run_path / "files" / "config.yaml")
  try:
    config = convert_wandb_style_config_to_omega_config(config)
  except Exception:
    pass
  with open_dict(config.data):
    config.data.setdefault("dac_model", "unidac4")
    config.data.setdefault("tps", 100)
    config.data.setdefault("lmx_vocab_path", "vocab/lmx_vocab_singletoken_asap.txt")
    config.data.setdefault("max_pt_x_len", config.data.max_seq_len["pt"])
    config.data.setdefault("out_pt_height_token", False)
    config.data.data_dir = "dummy"
    config.data.vq_model_dir = str(W.vq_models_dir())
    config.data.dac_model_dir = str(W.dac_models_dir())
  # Fine-tuned runs record finetune_params.finetune=True with the path of the
  # run they started from. Their checkpoints keep the *pre-training* vocabulary
  # layout (the fine-tuning script copied the base run's modality settings even
  # when the recipe lists a single task), so the token-layout fields are taken
  # from the base run's config; only the slicing settings stay the run's own.
  ft = config.get("finetune_params") or {}
  if ft.get("finetune") and ft.get("finetune_path"):
    base_run = next((seg for seg in reversed(Path(str(ft.finetune_path)).parts) if seg.startswith("run-")), None)
    if base_run and base_run != run_path.name:
      base_cfg = _base_run_config(base_run)
      layout_keys = ["in_modal_type", "out_modal_type", "modal_direction", "image_height", "image_compress_factor",
                     "max_seq_len", "lmx_vocab_path", "midi_max_shift", "max_pt_x_len", "out_pt_height_token",
                     "num_special_tokens", "n_codebook", "codebook_size", "vq_model", "dac_model", "tps"]
      with open_dict(config.data):
        for k in layout_keys:
          if k in base_cfg.data:
            config.data[k] = base_cfg.data[k]
          elif k in config.data and k in ("max_pt_x_len", "out_pt_height_token", "tps"):
            del config.data[k]   # fall back to the same defaults the base run gets
        config.data.setdefault("max_pt_x_len", config.data.max_seq_len["pt"])
        config.data.setdefault("out_pt_height_token", False)
        config.data.setdefault("tps", 100)
        config.data.setdefault("dac_model", "unidac4")
        config.data.pretrained_run = base_run
  with open_dict(config):
    if "finetune_params" not in config:
      config.finetune_params = {}
    config.finetune_params.finetune = False
  return config


def _base_run_config(base_run: str) -> OmegaConf:
  """Config of the run a fine-tuned checkpoint started from: from the models
  directory when present, otherwise just its config.yaml from the Hub."""
  cfg_path = W.resolve_run_path(base_run) / "files" / "config.yaml"
  if not cfg_path.exists():
    from huggingface_hub import hf_hub_download
    try:
      cfg_path = Path(hf_hub_download(W.HF_WEIGHTS_REPO, f"{base_run}/files/config.yaml",
                                      local_dir=str(W.models_dir()), token=W.hf_token()))
    except Exception as e:  # noqa: BLE001
      raise FileNotFoundError(
        f"config.yaml of the pre-training run {base_run} is needed to load this fine-tuned checkpoint "
        f"(expected at {W.models_dir() / base_run / 'files'}); download failed: {e}") from e
  cfg = OmegaConf.load(cfg_path)
  try:
    cfg = convert_wandb_style_config_to_omega_config(cfg)
  except Exception:
    pass
  return cfg


def build_idx_handlers(config) -> Tuple[TokenIdxHandler, TokenIdxHandler]:
  """Recreate the input/output vocabularies exactly as
  `MultimodalTokenDatasetMaker.get_vocab` does, without touching any dataset
  manifest."""
  data = config.data
  in_types = list(data.in_modal_type)
  out_types = list(data.out_modal_type)
  total = set(in_types + out_types)
  n_special = data.num_special_tokens

  vocabs = {}
  if "lmx" in total:
    lmx_path = Path(data.lmx_vocab_path)
    if not lmx_path.is_absolute():
      lmx_path = W.REPO_ROOT / lmx_path
    vocabs["lmx"] = LMXVocab(vocab_txt_fn=str(lmx_path), num_special_tokens=n_special)
  if "pt" in total:
    vocabs["pt"] = RVQVocab(
      data.codebook_size,
      num_special_tokens=n_special + 1,
      token_height=data.image_height // data.image_compress_factor,
      n_codebook=data.n_codebook,
    )
  if "dac" in total:
    vocabs["dac"] = RVQVocab(data.codebook_size, num_special_tokens=n_special, n_codebook=data.n_codebook)
  if "midi" in total:
    vocabs["midi"] = NoteEventTokenizer(
      max_shift_steps=data.midi_max_shift, tps=data.tps, max_length=data.max_seq_len["midi"] - 2
    )

  max_seq_len = dict(data.max_seq_len)
  in_handler = TokenIdxHandler({k: vocabs[k] for k in in_types}, max_seq_len, data.max_pt_x_len)
  if in_types != out_types:
    out_handler = TokenIdxHandler(
      {k: vocabs[k] for k in out_types}, max_seq_len, data.max_pt_x_len, out_pt_height_token=data.out_pt_height_token
    )
  else:
    out_handler = in_handler
  return in_handler, out_handler


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class SystemCrop:
  image: np.ndarray            # RGB uint8 crop
  page: int                    # 0-based page index
  index: int                   # 0-based system index within the page
  bbox: Tuple[int, int, int, int]
  conf: float

  @property
  def label(self) -> str:
    return f"page {self.page + 1} · system {self.index + 1} ({self.conf:.2f})"


@dataclass
class OMRResult:
  lmx_per_system: List[str]
  lmx: str
  musicxml: Optional[str]
  error: Optional[str] = None


@dataclass
class AudioResult:
  sample_rate: int
  audio: np.ndarray            # float32 mono
  n_tokens: int
  notes: List[str] = field(default_factory=list)

  @property
  def duration(self) -> float:
    return len(self.audio) / self.sample_rate


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #

class UMusTEngine:
  """One translation run plus the codecs it needs.

  `run_name` selects a run directory under the models directory (the default
  is the multi-task piano run); `run_path` overrides it with an explicit
  directory. Task-specific fine-tuned runs may declare fewer modalities (e.g.
  OMR: image -> notation only), so `supports(task)` tells which of the demo
  tasks a run can serve. Codecs can be shared between engines with
  `vq_model` / `dac_model`."""

  def __init__(self, run_path: Optional[Path] = None, device: Optional[str] = None, download: bool = True,
               run_name: Optional[str] = None, vq_model=None, dac_model=None):
    self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    self.run_name = run_name or W.PIANO_RUN
    if run_path is None:
      run_path = W.resolve_run_path(self.run_name)
    run_path = Path(run_path)
    if download:
      run_path, _, _ = W.ensure_all(self.run_name, config=load_run_config(run_path) if W.run_is_complete(run_path) else None)
    self.run_path = run_path
    self.config = load_run_config(run_path)

    self.in_handler, self.out_handler = build_idx_handlers(self.config)
    vq_emb, dac_emb = None, None
    if self.config.data.get("vq_model"):
      if vq_model is None:
        vq_model, vq_emb = get_vq_model(self.config)
      else:
        _, vq_emb = self._codebook_embeddings_vq(vq_model)
    else:
      vq_model = None
    if self.config.data.get("dac_model"):
      if dac_model is None:
        dac_model, dac_emb = get_dac_model(self.config)
      else:
        _, dac_emb = self._codebook_embeddings_dac(dac_model)
    else:
      dac_model = None
    dataset_ns = SimpleNamespace(in_idx_handler=self.in_handler, out_idx_handler=self.out_handler)
    model = get_model(self.config, vq_emb, dac_emb, dataset_ns)

    ckpts = [p for p in (run_path / "files" / "checkpoints").glob("*.pt") if p.name != "last_checkpoint.pt"]
    if not ckpts:
      raise FileNotFoundError(f"No checkpoint under {run_path}/files/checkpoints")
    ckpt_path = max(ckpts, key=W.checkpoint_iteration)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    load_model_state_dict(model, state)
    self.checkpoint_name = ckpt_path.name

    self.model = model.eval().to(self.device)
    self.vq_model = vq_model.eval().to(self.device) if vq_model is not None else None
    self.dac_model = dac_model.eval().to(self.device) if dac_model is not None else None
    self._yolo_system = None
    self._yolo_staff = None
    # The detectors are cheap; running them on CPU keeps them usable outside
    # GPU-decorated functions on ZeroGPU and off the model's GPU elsewhere.
    self.yolo_device = os.environ.get("UMUST_YOLO_DEVICE", "cpu")

    self.vq_version = self.config.data.get("vq_model") or "unirqvae3"
    self.target_staff_height = 18 if self.vq_version == "unirqvae3" else 20
    self.max_token_height = self.in_handler.max_token_height
    self.compress = self.config.data.image_compress_factor

    keys_in, keys_out = self.in_handler.vocab_keys, self.out_handler.vocab_keys
    def _index(keys, name):
      return keys.index(name) if name in keys else None
    self.idx = SimpleNamespace(
      pt_in=_index(keys_in, "pt"), midi_in=_index(keys_in, "midi"),
      lmx_out=_index(keys_out, "lmx"), dac_out=_index(keys_out, "dac"),
    )
    self.max_len = dict(self.config.data.max_seq_len)
    self.midi_tokenizer: Optional[NoteEventTokenizer] = self.in_handler.vocabs.get("midi")
    self.sep_token_id = self.in_handler.img_crop_cat_sep_idx if "pt" in keys_in else None
    # MIDI shift tokens cover (midi_max_shift - 1) / tps seconds; windows must stay inside
    self.midi_max_window_sec = (int(self.config.data.get("midi_max_shift", 2001)) - 1) / float(self.config.data.get("tps", 100))
    self.midi_input_codebooks = self._midi_input_codebooks()

  def _midi_input_codebooks(self) -> int:
    """How wide the encoder input rows were when this run saw MIDI.

    A MIDI token occupies one codebook, an image or audio token four, and the
    collate function pads every input of a batch to the widest one in it
    (`multimodal_collate_fn` in umust/data_utils.py). A run trained on MIDI
    together with score images therefore saw MIDI rows padded with three zeros,
    while a MIDI-only fine-tune saw one-column rows. The encoder embedding sums
    over the codebook dimension, so feeding the other width adds (or drops)
    three times the pad embedding on every token and the rendered performance
    drifts out of time with the score.

    The width follows from the datasets the run was trained on: the input side
    of each pair, resolved the way the dataset does it (`modal_direction`
    picks the modality earliest in the translation chain).
    """
    order = ("pt", "lmx", "midi", "dac")                          # 'omr' direction
    direction = str(self.config.data.get("modal_direction", "omr") or "omr")
    if direction == "amt":
      order = tuple(reversed(order))
    inputs = set()
    for entry in self.config.data.get("data_path") or []:
      kinds = [m for m in (entry[2] if len(entry) > 2 else []) if m in order]
      if not kinds:
        continue
      inputs.update(kinds if direction == "bi" else [min(kinds, key=order.index)])
    if not inputs:                                                # unknown recipe: the multi-task layout
      return int(self.config.data.n_codebook)
    return int(self.config.data.n_codebook) if inputs & {"pt", "dac"} else 1

  @staticmethod
  def _codebook_embeddings_vq(vq_model):
    """Codebook vectors used to initialize the token embeddings (as in
    umust.utils.get_vq_model), on CPU because the model is built there."""
    emb = {i: vq_model.quantizer.codebooks[i].weight.detach().cpu().clone()[:1024]
           for i in range(len(vq_model.quantizer.codebooks))}
    return vq_model, emb

  @staticmethod
  def _codebook_embeddings_dac(dac_model):
    emb = {}
    for i, q in enumerate(dac_model.quantizer.quantizers):
      weight = q.out_proj.weight.detach().cpu().clone()
      bias = q.out_proj.bias.detach().cpu().clone()
      codebook = q.codebook.weight.detach().cpu().clone()
      emb[i] = (weight.squeeze(-1) @ codebook.T).T + bias
    return dac_model, emb

  def supports(self, task: str) -> bool:
    """Which demo tasks this run can serve: omr, image_to_audio, contin_u, midi_to_audio."""
    has_pt, has_midi = self.idx.pt_in is not None, self.idx.midi_in is not None
    has_lmx, has_dac = self.idx.lmx_out is not None, self.idx.dac_out is not None and self.dac_model is not None
    return {
      "omr": has_pt and has_lmx and self.vq_model is not None,
      "image_to_audio": has_pt and has_dac and self.vq_model is not None,
      "contin_u": has_pt and has_dac and self.vq_model is not None,
      "midi_to_audio": has_midi and has_dac and self.midi_tokenizer is not None,
    }.get(task, False)

  def _require(self, task: str):
    if not self.supports(task):
      raise ValueError(f"run {self.run_name} ({self.checkpoint_name}) does not support {task}: "
                       f"inputs {self.in_handler.vocab_keys}, outputs {self.out_handler.vocab_keys}")

  # ----------------------------------------------------------------------- #
  # YOLO: system detection and staff-height estimation
  # ----------------------------------------------------------------------- #
  @property
  def yolo_system(self):
    if self._yolo_system is None:
      from ultralytics import YOLO
      self._yolo_system = YOLO(W.ensure_yolo("ls-yolo-system-v2.0.0.pt"))
    return self._yolo_system

  @property
  def yolo_staff(self):
    if self._yolo_staff is None:
      from ultralytics import YOLO
      self._yolo_staff = YOLO(W.ensure_yolo("ls-yolo-staff-height-v2.0.0.pt"))
    return self._yolo_staff

  def detect_systems(self, page_rgb: np.ndarray, page_index: int = 0, conf_threshold: float = 0.4) -> List[SystemCrop]:
    """Detect musical systems on one page image, sorted top-to-bottom."""
    results = self.yolo_system([page_rgb], verbose=False, device=self.yolo_device)
    crops = []
    for result in results:
      if len(result.boxes) == 0:
        continue
      boxes = result.boxes.xyxy.int().tolist()
      confs = result.boxes.conf.tolist()
      dets = sorted(zip(boxes, confs), key=lambda x: (x[0][1], x[0][0]))
      k = 0
      for (lx, ly, rx, ry), conf in dets:
        if conf < conf_threshold:
          continue
        crop = page_rgb[max(ly, 0):ry, max(lx, 0):rx]
        if crop.size == 0:
          continue
        crops.append(SystemCrop(crop, page_index, k, (lx, ly, rx, ry), conf))
        k += 1
    return crops

  def systems_from_images(self, pages: Sequence[np.ndarray], fallback_whole_image: bool = True) -> List[SystemCrop]:
    crops: List[SystemCrop] = []
    for i, page in enumerate(pages):
      page_crops = self.detect_systems(page, page_index=i)
      if not page_crops and fallback_whole_image:
        # already a system crop (or detector missed): use the whole image
        h, w = page.shape[:2]
        page_crops = [SystemCrop(page, i, 0, (0, 0, w, h), 0.0)]
      crops.extend(page_crops)
    return crops

  def estimate_staff_height(self, crop_rgb: np.ndarray) -> float:
    left_half = crop_rgb[:, : max(1, crop_rgb.shape[1] // 2)]
    if left_half.ndim == 2 or left_half.shape[2] == 1:
      left_half = cv2.cvtColor(left_half, cv2.COLOR_GRAY2RGB)
    results = self.yolo_staff([left_half], verbose=False, device=self.yolo_device)
    heights = []
    for result in results:
      boxes = result.boxes.xyxy.int().tolist()
      confs = result.boxes.conf.tolist()
      for (lx, ly, rx, ry), conf in zip(boxes, confs):
        if conf > 0.4:
          heights.append(ry - ly)
    if heights:
      return float(sum(heights) / len(heights))
    # fallback: assume a grand staff fills ~40% of the crop height
    return max(4.0, crop_rgb.shape[0] * 0.2)

  # ----------------------------------------------------------------------- #
  # image preprocessing + RQ-VAE tokenization
  # ----------------------------------------------------------------------- #
  def preprocess_system(self, crop_rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """Resize a system crop so its staff height matches the training data,
    whiten the background and pad to a multiple of the RQ-VAE patch size.
    Returns (normalized tensor [1,H,W], the resized 8-bit preview image)."""
    staff_height = self.estimate_staff_height(crop_rgb)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
    ratio = self.target_staff_height / staff_height
    # keep the token grid inside the positional-embedding range of the model
    max_h = self.max_token_height * self.compress - 3
    if gray.shape[0] * ratio > max_h:
      ratio = max_h / gray.shape[0]
    r_h, r_w = max(1, int(gray.shape[0] * ratio)), max(1, int(gray.shape[1] * ratio))
    resized = cv2.resize(gray, (r_w, r_h), interpolation=cv2.INTER_AREA)

    arr = resized.copy()
    median = np.median(arr)
    arr[arr > (median - 20)] = 255

    tensor = torchvision.transforms.functional.to_tensor(PIL.Image.fromarray(arr))
    h_pad = (16 - tensor.shape[-2] % 16) % 16
    w_pad = (16 - tensor.shape[-1] % 16) % 16
    tensor = torch.nn.functional.pad(tensor, (4, 3 + w_pad, 2, 1 + h_pad), mode="constant", value=1.0)
    tensor = torchvision.transforms.functional.normalize(tensor, mean=[0.5], std=[0.5])
    return tensor, arr

  @torch.no_grad()
  def tokenize_system(self, crop_rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    tensor, preview = self.preprocess_system(crop_rgb)
    codes = self.vq_model.get_codes(tensor.to(self.device).unsqueeze(0))
    codes = codes.to(torch.int16).cpu()
    if codes.ndim == 4:      # (1, h, w, n_codebook) -> (x_shift=1, y_shift=1, h, w, n_codebook)
      codes = codes.unsqueeze(0)
    return codes, preview

  # ----------------------------------------------------------------------- #
  # low-level inference helpers
  # ----------------------------------------------------------------------- #
  def _pt_batch(self, pt_list: List[torch.Tensor]):
    data, token_height, pos = self.in_handler(list(pt_list), "pt", add_height_token=False)
    in_modal = data.squeeze(0).squeeze(0) if data.ndim == 4 else data  # (T, n_codebook)
    in_modal = in_modal.reshape(-1, in_modal.shape[-1]).unsqueeze(0).long()
    in_pos = pos.unsqueeze(0).long()
    in_mask = torch.ones(in_modal.shape[:2], dtype=torch.bool)
    return in_modal.to(self.device), in_pos.to(self.device), in_mask.to(self.device), int(token_height)

  def _modal_idx(self, in_idx: int, out_idx: int) -> torch.Tensor:
    return torch.tensor([[in_idx, out_idx]], dtype=torch.long, device=self.device)

  @torch.no_grad()
  def _generate(self, in_modal, in_pos, in_mask, modal_idx, token_height, out_key: str,
                condition=None, seed: int = 0, sampling_method=None, threshold=0.9, temperature=1.0,
                max_length: Optional[int] = None):
    token_heights = torch.tensor([[token_height, 0]], dtype=torch.int16, device=self.device)
    return self.model.inference(
      in_modal=in_modal, in_pos=in_pos, modal_idx=modal_idx, in_mask=in_mask,
      token_heights=token_heights, sampling_method=sampling_method, threshold=threshold,
      temperature=temperature, manual_seed=seed,
      max_length=max_length or self.max_len[out_key], condition=condition,
    )

  def _strip_dac(self, tokens: torch.Tensor) -> torch.Tensor:
    """Shifted output tokens (1, T, n_codebook) -> raw DAC codes (1, n_codebook, T)."""
    tokens = tokens.detach().cpu()
    if tokens.ndim == 2:
      tokens = tokens.unsqueeze(0)
    eos = self.out_handler.shifted_eos_tensors[self.idx.dac_out]        # (1, n_codebook)
    sos = self.out_handler.shifted_sos_tensors[self.idx.dac_out]
    if (tokens[:, 0] == sos).all():
      tokens = tokens[:, 1:]
    is_eos = (tokens == eos).all(dim=-1)[0]
    if is_eos.any():
      tokens = tokens[:, : int(is_eos.nonzero()[0])]
    codes = tokens - self.out_handler.idx_shifts["dac"] - self.out_handler.rq_shifter["dac"]
    codes = codes.clamp(min=0).long()
    return codes.permute(0, 2, 1)

  def _truncate_generated(self, tokens: torch.Tensor) -> torch.Tensor:
    """Drop everything from the first EOS on (keeps SOS/condition prefix).

    With a single special token per modality SOS == EOS, so position 0 (the
    start token) is excluded from the search, as the decoder itself does."""
    eos = self.out_handler.shifted_eos_tensors[self.idx.dac_out]
    is_eos = (tokens[:, 1:].detach().cpu() == eos).all(dim=-1)[0]
    if is_eos.any():
      return tokens[:, : int(is_eos.nonzero()[0]) + 1]
    return tokens

  @torch.no_grad()
  def decode_audio(self, tokens: torch.Tensor) -> AudioResult:
    codes = self._strip_dac(tokens).to(self.device)
    n_tokens = codes.shape[-1]
    if n_tokens == 0:
      return AudioResult(int(self.dac_model.sample_rate), np.zeros(1, dtype=np.float32), 0, ["model produced no audio tokens"])
    # Decode in roughly equal chunks so that no chunk is shorter than the
    # decoder's receptive field (DAC decodes chunks independently).
    n_chunks = max(1, math.ceil(n_tokens / 4096))
    chunk = math.ceil(n_tokens / n_chunks)
    signal = self.dac_model.decompress_tensor(codes, n_quantizers=self.config.data.n_codebook, chunk_length=chunk)
    audio = signal.audio_data.squeeze().float().cpu().numpy()
    if audio.ndim > 1:
      audio = audio.mean(axis=0)
    return AudioResult(int(signal.sample_rate), audio.astype(np.float32), n_tokens)

  # ----------------------------------------------------------------------- #
  # task 1: OMR
  # ----------------------------------------------------------------------- #
  @torch.no_grad()
  def omr_system(self, crop_rgb: np.ndarray, greedy: bool = True, temperature: float = 0.1, seed: int = 0) -> str:
    self._require("omr")
    pt, _ = self.tokenize_system(crop_rgb)
    in_modal, in_pos, in_mask, height = self._pt_batch([pt])
    out = self._generate(
      in_modal, in_pos, in_mask, self._modal_idx(self.idx.pt_in, self.idx.lmx_out), height, "lmx",
      seed=seed, sampling_method="argmax" if greedy else None, temperature=temperature,
    )
    ids = out[0, :, 0].detach().cpu() - self.out_handler.idx_shifts["lmx"]
    ids[ids < 0] = 0
    return self.out_handler.vocabs["lmx"].decode(ids)

  def lmx_to_musicxml(self, lmx: str) -> Tuple[Optional[str], Optional[str]]:
    try:
      return delinearize_lmx(lmx), None
    except Exception as e:  # noqa: BLE001 - surface any delinearizer failure to the UI
      return None, f"{type(e).__name__}: {e}"

  def omr(self, systems: Sequence[SystemCrop], greedy: bool = True, temperature: float = 0.1, seed: int = 0,
          progress: ProgressFn = _noop_progress) -> OMRResult:
    lmx_list = []
    for i, s in enumerate(systems):
      progress(i / max(1, len(systems)), f"OMR system {i + 1}/{len(systems)}")
      lmx_list.append(self.omr_system(s.image, greedy=greedy, temperature=temperature, seed=seed))
    lmx = " ".join(x for x in lmx_list if x)
    xml, err = self.lmx_to_musicxml(lmx) if lmx else (None, "empty LMX output")
    progress(1.0, "done")
    return OMRResult(lmx_list, lmx, xml, err)

  # ----------------------------------------------------------------------- #
  # task 2: image -> audio (one window, 1..N systems concatenated with [SEP])
  # ----------------------------------------------------------------------- #
  @torch.no_grad()
  def image_to_audio(self, systems: Sequence[SystemCrop], seed: int = 0,
                     progress: ProgressFn = _noop_progress) -> AudioResult:
    self._require("image_to_audio")
    if len(systems) == 0:
      raise ValueError("no system images given")
    if len(systems) > 2:
      return self.contin_u(systems, seed=seed, progress=progress)
    progress(0.05, "tokenizing")
    pts = [self.tokenize_system(s.image)[0] for s in systems]
    in_modal, in_pos, in_mask, height = self._pt_batch(pts)
    progress(0.2, "generating audio tokens")
    out = self._generate(
      in_modal, in_pos, in_mask, self._modal_idx(self.idx.pt_in, self.idx.dac_out), height, "dac",
      seed=seed, sampling_method="none", threshold=0.9, temperature=1.0,
    )
    progress(0.9, "decoding audio")
    result = self.decode_audio(out)
    progress(1.0, "done")
    return result

  # ----------------------------------------------------------------------- #
  # task 3: Contin-U (two-system sliding window, ported from infer.py)
  # ----------------------------------------------------------------------- #
  @torch.no_grad()
  def contin_u(self, systems: Sequence[SystemCrop], seed: int = 0, attn_threshold: float = 0.5,
               min_run: int = 3, progress: ProgressFn = _noop_progress) -> AudioResult:
    self._require("contin_u")
    n = len(systems)
    if n == 0:
      raise ValueError("no system images given")
    progress(0.0, f"tokenizing {n} systems")
    pts = [self.tokenize_system(s.image)[0] for s in systems]
    if n == 1:
      in_modal, in_pos, in_mask, height = self._pt_batch(pts)
      out = self._generate(in_modal, in_pos, in_mask, self._modal_idx(self.idx.pt_in, self.idx.dac_out), height, "dac",
                           seed=seed, sampling_method="none", threshold=0.9, temperature=1.0)
      return self.decode_audio(out)

    modal_idx = self._modal_idx(self.idx.pt_in, self.idx.dac_out)
    attend = self.model.decoder.net.decoder.transformer_decoder.layers[-2][-2].attend
    notes = []
    condition = None
    out_tokens = []
    for j in range(n - 1):
      progress(0.05 + 0.85 * j / (n - 1), f"window {j + 1}/{n - 1}: systems {j + 1}+{j + 2}")
      in_modal, in_pos, in_mask, height = self._pt_batch(pts[j:j + 2])
      peeper = LayerPeeper(attend, hook_fn=use_attn_weights)
      try:
        out = self._generate(in_modal, in_pos, in_mask, modal_idx, height, "dac", condition=condition,
                             seed=seed, sampling_method="none", threshold=0.9, temperature=1.0)
      finally:
        peeper.remove()

      if j != 0:
        hooked = peeper.output[1:]
        hooked = [peeper.output[0][:, :, k:k + 1] for k in range(peeper.output[0].size(2))] + hooked
      else:
        hooked = peeper.output
      attn = torch.stack(hooked).squeeze(-2).squeeze(1)          # (T, heads, K)
      sep_idx = int((in_modal == self.sep_token_id).nonzero(as_tuple=True)[1][0])
      heads = torch.stack([attn[:, 0], attn[:, 1], attn[:, 7], attn[:, 10]], dim=1).cpu()
      border = find_audio_border(heads, sep_idx=sep_idx, thr=attn_threshold, min_run=min_run)
      if border < 0:
        notes.append(f"window {j + 1}: no attention boundary found; using sequence end")
        border = out.size(1) - 1
      border = max(border - 10, 1)
      cond_length = max((out.size(1) - border) // 5, 1)

      if j == 0:
        out_tokens.append(out[:, :border])
      elif j == n - 2:
        out_tokens.append(out[:, 1:border])
        out_tokens.append(out[:, border:])
      else:
        out_tokens.append(out[:, 1:border])
      if j == 0 and n == 2:
        out_tokens.append(out[:, border:])
      condition = out[:, border:border + cond_length]

    final = torch.cat(out_tokens, dim=1)
    progress(0.95, "decoding audio")
    result = self.decode_audio(final)
    result.notes = notes
    progress(1.0, "done")
    return result

  # ----------------------------------------------------------------------- #
  # task 4: MIDI -> audio (windowed, audio-prefix conditioned)
  # ----------------------------------------------------------------------- #
  def midi_window_limit_sec(self) -> float:
    """Longest MIDI window this run can render: the decoder's audio-token
    budget (minus SOS/EOS margin) and the MIDI shift-token range, whichever is
    shorter (20.2 s for the released piano runs)."""
    budget = (self.max_len["dac"] - 8) / DAC_TOKENS_PER_SEC
    return float(min(budget, self.midi_max_window_sec))

  def midi_window_default_sec(self, overlap_sec: float = 2.0) -> float:
    """Window length matching the slices the run was trained on
    (`midi_slice_len`, minus the random margin used in training), kept within
    the decoder budget together with the overlap prefix: 18 s for the
    multi-task run (20 s slices), 9.5 s for the MIDI-to-audio run fine-tuned
    on 10 s slices."""
    slice_len = float(self.config.data.get("midi_slice_len", 20) or 20)
    return float(max(2.0, min(slice_len - 0.5, self.midi_window_limit_sec() - overlap_sec)))

  def load_midi_notes(self, midi_path: str):
    notes, duration = midi2note(
      midi_path, binary_velocity=True, ch_9_as_drum=False, force_all_drum=False, force_all_program_to=0,
      trim_overlap=True, fix_offset=True, quantize=True, verbose=0, minimum_offset_sec=0.01,
      drum_offset_sec=0.01, ignore_pedal=False,
    )
    note_events = note2note_event(notes)
    return notes, note_events, float(duration)

  def _encode_midi_window(self, note_events, start: float, end: float) -> List[int]:
    sliced, tied, start_time = slice_note_events_and_ties(note_events, start, end)
    return self.midi_tokenizer.encode(sliced, tie_note_events=tied, start_time=start_time, end_time=end)

  def _fit_midi_window(self, note_events, start: float, end: float, hard_end: float):
    """Shrink [start, end) until the MIDI tokens fit the encoder budget."""
    budget = self.max_len["midi"] - 2
    while True:
      tokens = self._encode_midi_window(note_events, start, end)
      if len(tokens) <= budget or end - start <= 1.0:
        return tokens, end
      end = max(start + 1.0, end - 1.0)

  def _midi_batch(self, tokens: List[int]):
    data, _, pos = self.in_handler(tokens, "midi")                # (T, 1) shifted, (T, 2)
    n_cb = self.midi_input_codebooks                              # as in this run's training batches
    if n_cb > data.shape[1]:
      data = torch.nn.functional.pad(data, (0, n_cb - data.shape[1]))
    in_modal = data.unsqueeze(0).long().to(self.device)
    in_pos = pos.unsqueeze(0).long().to(self.device)
    in_mask = torch.ones(in_modal.shape[:2], dtype=torch.bool, device=self.device)
    return in_modal, in_pos, in_mask

  @torch.no_grad()
  def midi_to_audio(self, midi_path: str, window_sec: float = 0.0, overlap_sec: float = 2.0,
                    max_duration_sec: Optional[float] = None, seed: int = 0,
                    progress: ProgressFn = _noop_progress) -> AudioResult:
    """Render a performance MIDI file with overlapping windows.

    The model was trained on 19-20 s MIDI/audio slices and keeps producing
    audio for about that long whatever the window covers, so each window's
    output is cut to the duration of its MIDI content (MIDI-to-audio is
    time-aligned: 86.13 token sets per second). The audio tokens already
    generated for the start of the next window are fed back as the decoder
    prefix (`condition`), which keeps the performance continuous across the
    seams. A short final window is extended backwards to a full-length one.
    """
    self._require("midi_to_audio")
    notes, note_events, duration = self.load_midi_notes(midi_path)
    if not notes:
      raise ValueError("the MIDI file contains no notes")
    rate = DAC_TOKENS_PER_SEC
    max_window = self.midi_window_limit_sec()
    if not window_sec or window_sec <= 0:                      # 0 = the run's native window
      window_sec = self.midi_window_default_sec(overlap_sec)
    window_sec = float(min(max(window_sec, 1.0), max_window))
    overlap_sec = float(min(max(overlap_sec, 0.0), window_sec / 2))
    total = duration if max_duration_sec is None else min(duration, float(max_duration_sec))

    modal_idx = self._modal_idx(self.idx.midi_in, self.idx.dac_out)
    stream: Optional[torch.Tensor] = None                        # (1, T, n_cb), aligned to t = 0
    notes_out = []
    start, w = 0.0, 0
    while True:
      end = min(start + window_sec, total)
      if end - start < window_sec and start > 0:                 # last, short window: extend backwards
        start = max(0.0, total - window_sec)
      tokens, end = self._fit_midi_window(note_events, start, end, total)
      progress(min(0.9, start / max(total, 1e-3)), f"window {w + 1}: {start:.1f}-{end:.1f}s ({len(tokens)} MIDI tokens)")

      n_cond = 0
      condition = None
      if stream is not None:
        n_cond = stream.shape[1] - int(round(start * rate))
        if n_cond > 0:
          condition = stream[:, -n_cond:]
        else:
          n_cond = 0
      in_modal, in_pos, in_mask = self._midi_batch(tokens)
      out = self._generate(in_modal, in_pos, in_mask, modal_idx, 0, "dac", condition=condition,
                           seed=seed + w, sampling_method="none", threshold=0.9, temperature=1.0)
      body = self._truncate_generated(out)[:, 1:]                 # drop SOS, EOS and padding
      expected = int(round((end - start) * rate))
      if body.shape[1] < expected:
        # the model closed the window early; keep the stream time-aligned by
        # treating the covered span as this window's end
        covered_end = start + body.shape[1] / rate
        notes_out.append(f"window {w + 1}: model stopped after {body.shape[1] / rate:.1f}s of {end - start:.1f}s")
        if end < total - 1e-3:
          end = covered_end
      body = body[:, :expected]
      new_tokens = body[:, n_cond:]
      stream = new_tokens if stream is None else torch.cat([stream, new_tokens], dim=1)

      if end >= total - 1e-3:
        break
      start = end - overlap_sec
      w += 1

    sos = self.out_handler.shifted_sos_tensors[self.idx.dac_out].view(1, 1, -1).to(stream.device, stream.dtype)
    final = torch.cat([sos, stream], dim=1)
    progress(0.95, "decoding audio")
    result = self.decode_audio(final)
    result.notes = notes_out + [f"{w + 1} window(s), {len(notes)} notes, MIDI duration {duration:.1f}s"]
    progress(1.0, "done")
    return result


# --------------------------------------------------------------------------- #
# helpers shared with infer.py
# --------------------------------------------------------------------------- #

def find_audio_border(attn: torch.Tensor, sep_idx: int, thr: float = 0.5, min_run: int = 3) -> int:
  """First audio step whose (head-averaged) attention mass on the tokens after
  [SEP] stays above `thr` for `min_run` consecutive steps; -1 if none."""
  head_avg = attn.mean(dim=1)
  post_sep_share = head_avg[:, sep_idx + 1:].sum(dim=1)
  over = (post_sep_share >= thr).float()
  run = torch.nn.functional.conv1d(over[None, None, :], weight=torch.ones(1, 1, min_run), padding=min_run - 1)[0, 0]
  idx = (run >= min_run).nonzero(as_tuple=True)[0]
  return int(idx[0]) if idx.numel() else -1


# --------------------------------------------------------------------------- #
# document loading
# --------------------------------------------------------------------------- #

def pdf_to_page_images(pdf_path: str, dpi: int = 300, first_page: int = 1, last_page: Optional[int] = None) -> List[np.ndarray]:
  """Rasterize PDF pages (1-based inclusive range) to RGB arrays."""
  import pymupdf
  doc = pymupdf.open(pdf_path)
  n = len(doc)
  first = max(1, first_page)
  last = n if last_page is None else min(n, last_page)
  pages = []
  for i in range(first - 1, last):
    pix = doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    pages.append(arr.copy())
  doc.close()
  return pages


def find_musescore() -> Optional[str]:
  import shutil, os
  env = os.environ.get("MSCORE_PATH")
  candidates = [env, str(W.REPO_ROOT / "mscore-3.6.2" / "AppRun"), "/app/mscore-3.6.2/AppRun",
                shutil.which("mscore3"), shutil.which("musescore3"), shutil.which("mscore"), shutil.which("musescore")]
  for c in candidates:
    if c and Path(c).exists():
      return c
  return None


def musicxml_to_pdf(mxl_path: str, out_pdf: Path, mscore_path: str) -> Path:
  """Engrave MusicXML with MuseScore 3.6.2 (needs an X display; uses xvfb-run when available)."""
  import os, shutil, subprocess
  cmd = [mscore_path, "-n", "-r", "300", "-o", str(out_pdf), str(mxl_path)]
  env = os.environ.copy()
  env.setdefault("QT_QPA_PLATFORM", "xcb")
  if shutil.which("xvfb-run") and not env.get("DISPLAY"):
    cmd = ["xvfb-run", "-a"] + cmd
  ps = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
  if ps.returncode != 0 or not out_pdf.exists():
    raise RuntimeError(f"MuseScore failed ({ps.returncode}): {ps.stderr.decode('utf-8', 'ignore')[-800:]}")
  return out_pdf


def load_image_rgb(path: str) -> np.ndarray:
  img = PIL.Image.open(path).convert("RGB")
  return np.array(img)


def piano_roll_image(midi_path: str, max_sec: Optional[float] = None) -> np.ndarray:
  """Render a simple piano roll of a MIDI file as an RGB array."""
  import pretty_midi
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  pm = pretty_midi.PrettyMIDI(midi_path)
  fs = 50
  roll = pm.get_piano_roll(fs=fs)
  if max_sec is not None:
    roll = roll[:, : int(max_sec * fs)]
  lo, hi = 21, 109
  fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
  ax.imshow(roll[lo:hi] > 0, aspect="auto", origin="lower", cmap="magma",
            extent=[0, roll.shape[1] / fs, lo, hi], interpolation="nearest")
  ax.set_xlabel("time (s)")
  ax.set_ylabel("MIDI pitch")
  fig.tight_layout()
  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return buf


def render_musicxml_svgs(xml: str, layout: str = "page", scale: int = 40) -> List[str]:
  """Engrave MusicXML with Verovio (pure pip dependency) and return one SVG
  document per page. `layout="system"` puts the whole content on a single
  line (one system), which mirrors the system crops the OMR model reads;
  `layout="page"` breaks it into A4-proportioned pages."""
  try:
    import verovio
    from importlib.resources import files as _files
  except ImportError:
    return []
  # Verovio keeps its *default* resource path in thread-local storage, so a
  # toolkit created in a worker thread (Gradio runs handlers in a thread pool)
  # silently fails to load its fonts. Set the path on the instance instead.
  tk = verovio.toolkit(False)
  if not tk.setResourcePath(str(_files("verovio") / "data")):
    return []
  options = {
    "scale": scale, "adjustPageHeight": True, "footer": "none", "header": "none", "svgViewBox": True,
    "pageMarginLeft": 40, "pageMarginRight": 40, "pageMarginTop": 40, "pageMarginBottom": 40,
  }
  if layout == "system":
    options.update({"breaks": "none", "adjustPageWidth": True, "pageWidth": 60000})
  else:
    options.update({"breaks": "auto", "pageWidth": 2100, "spacingSystem": 8})
  tk.setOptions(options)
  if not tk.loadData(xml):
    return []
  return [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]


def render_musicxml_svg(xml: str, scale: int = 35) -> Optional[str]:
  """Backward-compatible helper: all pages joined into one HTML fragment."""
  svgs = render_musicxml_svgs(xml, layout="page", scale=scale)
  return "\n".join(svgs) if svgs else None


def svg_to_image(svg: str, width: int = 2400) -> Optional[np.ndarray]:
  """Rasterize an SVG document to an RGB array (resvg, pure pip); None if unavailable."""
  try:
    import resvg_py
  except ImportError:
    return None
  import io
  try:
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, background="white"))
  except Exception:  # noqa: BLE001 - malformed SVG or renderer failure
    return None
  return np.array(PIL.Image.open(io.BytesIO(png)).convert("RGB"))


def render_lmx_image(lmx: str, layout: str = "system", width: int = 2400) -> Tuple[Optional[np.ndarray], Optional[str]]:
  """LMX string -> engraved RGB image (first page/system). Returns (image, error)."""
  if not lmx.strip():
    return None, "empty LMX"
  try:
    xml = delinearize_lmx(lmx)
  except Exception as e:  # noqa: BLE001
    return None, f"{type(e).__name__}: {e}"
  svgs = render_musicxml_svgs(xml, layout=layout)
  if not svgs:
    return None, "Verovio could not engrave this MusicXML"
  img = svg_to_image(svgs[0], width=width)
  return img, (None if img is not None else "SVG rasterization unavailable")
