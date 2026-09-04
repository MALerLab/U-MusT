#!/usr/bin/env python3
"""Assemble (and optionally push) the Hugging Face Space for the demo.

    python demo/build_space.py --out build/space                # assemble only
    python demo/build_space.py --push malerlab/u-must-demo      # assemble + upload
    python demo/build_space.py --push <user>/u-must-demo --hardware zero-a10g --secret-token $TOKEN_WITH_WEIGHT_ACCESS

The Space bundles `app.py`, the `demo/` package, the model code (`umust/`,
`rqvae/`), the LMX vocabularies and the example files. Weights are *not*
bundled: the app downloads them at start-up, so the Space needs an `HF_TOKEN`
secret that has been granted access to the gated malerlab/u-must repository.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_TEMPLATE = ROOT / "demo" / "space"

CODE_DIRS = ["umust", "rqvae", "vocab"]
CODE_FILES = ["app.py", "LICENSE"]
DEMO_FILES = ["__init__.py", "engine.py", "weights.py"]
EXAMPLE_GLOB = ["*.pdf", "*.mid", "*.png"]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.pth", "*.ckpt", ".DS_Store")


def assemble(out: Path) -> Path:
  if out.exists():
    shutil.rmtree(out)
  out.mkdir(parents=True)
  for d in CODE_DIRS:
    shutil.copytree(ROOT / d, out / d, ignore=IGNORE)
  for f in CODE_FILES:
    shutil.copy2(ROOT / f, out / f)
  (out / "demo").mkdir()
  for f in DEMO_FILES:
    shutil.copy2(ROOT / "demo" / f, out / "demo" / f)
  (out / "demo" / "examples").mkdir()
  for pattern in EXAMPLE_GLOB:
    for f in (ROOT / "demo" / "examples").glob(pattern):
      shutil.copy2(f, out / "demo" / "examples" / f.name)
  for name in ["README.md", "requirements.txt", "packages.txt"]:
    shutil.copy2(SPACE_TEMPLATE / name, out / name)
  (out / ".gitattributes").write_text("*.pdf filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n*.mid filter=lfs diff=lfs merge=lfs -text\n")
  (out / ".gitignore").write_text("__pycache__/\nmodels/\nvq_models/\ndac_models/\nyolo/\n")
  print(f"Space assembled at {out}")
  return out


def push(folder: Path, repo_id: str, hardware: str | None, secret_token: str | None, private: bool):
  from huggingface_hub import HfApi
  api = HfApi()
  api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=private,
                  space_hardware=hardware)
  if secret_token:
    api.add_space_secret(repo_id, "HF_TOKEN", secret_token)
  api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="space",
                    commit_message="Update U-MusT demo")
  print(f"Pushed to https://huggingface.co/spaces/{repo_id}")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--out", default=str(ROOT / "build" / "space"), help="assembly directory")
  p.add_argument("--push", metavar="REPO_ID", help="Space id to upload to, e.g. malerlab/u-must-demo")
  p.add_argument("--hardware", default=None, help="Space hardware on creation, e.g. t4-small, a10g-small, zero-a10g")
  p.add_argument("--secret-token", default=os.environ.get("UMUST_SPACE_HF_TOKEN"),
                 help="token with access to the gated weights, stored as the Space's HF_TOKEN secret")
  p.add_argument("--private", action="store_true")
  args = p.parse_args()

  folder = assemble(Path(args.out))
  if args.push:
    push(folder, args.push, args.hardware, args.secret_token, args.private)


if __name__ == "__main__":
  sys.exit(main())
