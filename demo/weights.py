"""Locate (and, when missing, download) every checkpoint the demo needs.

Layout, relative to the repository root unless overridden by environment
variables:

    models/<run>/files/config.yaml, files/checkpoints/*.pt   UMUST_RUN_PATH
    vq_models/<vq_model>_f16_c1024_k4/                        UMUST_VQ_MODEL_DIR
    dac_models/<dac_model>/weights.pth                        UMUST_DAC_MODEL_DIR
    yolo/ls-yolo-*.pt                                         UMUST_YOLO_DIR

The translation weights are gated on the Hub, so downloading them needs a
token (`HF_TOKEN`) whose owner has been granted access; `UMUST_HF_WEIGHTS_REPO`
selects another repository with the same layout. The codecs and YOLO
detectors are public.
"""
import os
import re
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

PIANO_RUN = "run-20250225_062905-9n1554as"          # multi-task I2A piano run (OMR + M2A + I2A)
# Task-specific fine-tuned runs (50k further steps at 1e-5, paper Sec. VII-B),
# selected by the demo through UMUST_RUN_OMR / UMUST_RUN_MIDI; the multi-task
# run stays the default for every task until they are configured.
OMR_RUN = os.environ.get("UMUST_RUN_OMR", "")
MIDI_RUN = os.environ.get("UMUST_RUN_MIDI", "")
# Hub repository holding the translation weights. Override with
# UMUST_HF_WEIGHTS_REPO to point a Space at a mirror the deploying account can
# read (the run directory layout must be the same).
HF_WEIGHTS_REPO = os.environ.get("UMUST_HF_WEIGHTS_REPO", "malerlab/u-must")
HF_CODEC_REPOS = {
  "unirqvae3": "malerlab/unirqvae3-ytsv",
  "unirqvae": "malerlab/unirqvae-ytsv",
  "unidac4": "malerlab/unidac4-ytsv",
}
YOLO_MODELS_URLS = {
  "ls-yolo-system-v2.0.0.pt": "https://github.com/MALerLab/ls-yolo/releases/download/system-v2/ls-yolo-system-v2.0.0.pt",
  "ls-yolo-staff-height-v2.0.0.pt": "https://github.com/MALerLab/ls-yolo/releases/download/staff-height-v2/ls-yolo-staff-height-v2.0.0.pt",
}


def _env_path(name: str, default: Path) -> Path:
  value = os.environ.get(name)
  return Path(value).expanduser().resolve() if value else default


def models_dir() -> Path:
  return _env_path("UMUST_MODELS_DIR", REPO_ROOT / "models")


def vq_models_dir() -> Path:
  return _env_path("UMUST_VQ_MODEL_DIR", REPO_ROOT / "vq_models")


def dac_models_dir() -> Path:
  return _env_path("UMUST_DAC_MODEL_DIR", REPO_ROOT / "dac_models")


def yolo_dir() -> Path:
  return _env_path("UMUST_YOLO_DIR", REPO_ROOT / "yolo")


def hf_token() -> Optional[str]:
  return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def resolve_run_path(run_name: str = PIANO_RUN) -> Path:
  """Return the run directory holding `files/config.yaml`.

  Accepts both `models/<run>/files/...` and the `models/<run>/<run>/files/...`
  layout produced by unzipping the archived release."""
  explicit = os.environ.get("UMUST_RUN_PATH")
  if explicit and run_name == PIANO_RUN:
    return Path(explicit).expanduser().resolve()
  base = models_dir() / run_name
  nested = base / run_name
  if (nested / "files" / "config.yaml").exists():
    return nested
  return base


def run_is_complete(run_path: Path) -> bool:
  ckpt_dir = run_path / "files" / "checkpoints"
  return (run_path / "files" / "config.yaml").exists() and any(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else False


def download_run(run_name: str = PIANO_RUN) -> Path:
  """Fetch the translation weights from the gated Hub repository."""
  from huggingface_hub import snapshot_download
  token = hf_token()
  if token is None:
    raise RuntimeError(
      f"Translation weights not found under {models_dir() / run_name} and no HF_TOKEN is set. "
      f"Request access to https://huggingface.co/{HF_WEIGHTS_REPO}, then either download the run "
      f"there or set HF_TOKEN so the demo can fetch it."
    )
  snapshot_download(
    repo_id=HF_WEIGHTS_REPO,
    allow_patterns=[f"{run_name}/*"],
    local_dir=str(models_dir()),
    token=token,
  )
  return resolve_run_path(run_name)


def download_codec(name: str, target_dir: Path):
  from huggingface_hub import snapshot_download
  repo_id = HF_CODEC_REPOS[name]
  snapshot_download(repo_id=repo_id, local_dir=str(target_dir), token=hf_token())


def ensure_yolo(name: str) -> Path:
  path = yolo_dir() / name
  if not path.exists():
    url = YOLO_MODELS_URLS[name]
    print(f"Downloading YOLO checkpoint {name} from {url} ...")
    r = requests.get(url, allow_redirects=True, timeout=600)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
  return path


def vq_model_name_from_config(config) -> str:
  return f"{config.data.vq_model}_f{config.data.image_compress_factor}_c{config.data.codebook_size}_k{config.data.n_codebook}"


def ensure_all(run_name: str = PIANO_RUN, config=None):
  """Make sure the run, its codecs and both YOLO detectors are on disk.

  Returns (run_path, vq_models_dir, dac_models_dir). `config` may be passed to
  avoid loading the run config twice; when None it is read from the run."""
  run_path = resolve_run_path(run_name)
  if not run_is_complete(run_path):
    run_path = download_run(run_name)
  if config is None:
    from omegaconf import OmegaConf
    from umust.utils import convert_wandb_style_config_to_omega_config
    config = OmegaConf.load(run_path / "files" / "config.yaml")
    try:
      config = convert_wandb_style_config_to_omega_config(config)
    except Exception:
      pass

  vq_dir = vq_models_dir() / vq_model_name_from_config(config)
  if not any(vq_dir.rglob("*.pt")):
    download_codec(config.data.vq_model, vq_dir)
  dac_name = config.data.get("dac_model", "unidac4") or "unidac4"
  dac_dir = dac_models_dir() / dac_name
  if not (dac_dir / "weights.pth").exists():
    download_codec(dac_name, dac_dir)

  for name in YOLO_MODELS_URLS:
    ensure_yolo(name)
  return run_path, vq_models_dir(), dac_models_dir()


def checkpoint_iteration(path: Path) -> int:
  m = re.match(r"iter(\d+)", path.stem)
  return int(m.group(1)) if m else -1
