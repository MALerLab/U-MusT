#!/usr/bin/env python3
"""Set the description and links of the four Replicate models.

    REPLICATE_API_TOKEN=r8_... python replicate_models/set_metadata.py [owner]

The longer READMEs in replicate_models/readme/*.md are not settable through
the API; paste them into each model's README on replicate.com.
"""
import os
import sys

import requests

OWNER = sys.argv[1] if len(sys.argv) > 1 else "malerlab"
TOKEN = os.environ["REPLICATE_API_TOKEN"]
COMMON = {
  "github_url": "https://github.com/MALerLab/U-MusT",
  "weights_url": "https://huggingface.co/malerlab/u-must",
  "paper_url": "https://doi.org/10.1109/TASLPRO.2025.3648794",
  "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}
DESCRIPTIONS = {
  "u-must-omr": "Optical music recognition for piano scores: a page or system image → Linearized MusicXML and MusicXML, engraved back to images (U-MusT, IEEE TASLP 2026).",
  "u-must-midi-to-audio": "Piano audio synthesized from a performance MIDI file by the U-MusT translation model, with a plain soundfont rendering returned for comparison.",
  "u-must-image-to-audio": "Direct score-image-to-audio: a piano score page or system → performance audio with no symbolic step in between (U-MusT, IEEE TASLP 2026).",
  "u-must-contin-u": "Contin-U: a whole PDF piano score → one continuous performance, via the two-system sliding window of the U-MusT image-to-audio model (RenCon 2025 / MIREX).",
}
PAPER = {"u-must-contin-u": "https://futuremirex.com/portal/wp-content/uploads/2025/rencon/Contin-U.pdf"}

for name, description in DESCRIPTIONS.items():
  body = {**COMMON, "description": description}
  if name in PAPER:
    body["paper_url"] = PAPER[name]
  r = requests.patch(f"https://api.replicate.com/v1/models/{OWNER}/{name}",
                     headers={"Authorization": f"Bearer {TOKEN}"}, json=body)
  d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
  print(f"{OWNER}/{name}: HTTP {r.status_code} | description set: {d.get('description') == description} | paper: {d.get('paper_url')} | weights: {d.get('weights_url')}")
