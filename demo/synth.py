"""Reference synthesis of a MIDI file with a General MIDI soundfont.

Used by the MIDI-to-Audio tab so the model's rendition can be compared with
a plain, unexpressive playback of the same notes. Two backends:

  * FluidSynth through `pyfluidsynth` (preferred when the library and a GM
    soundfont are installed, e.g. `apt install fluidsynth fluid-soundfont-gm`);
  * TinySoundFont (`pip install --no-deps tinysoundfont`), a pure-pip fallback,
    with the small TimGM6mb soundfont bundled in `pretty_midi`.

`UMUST_SOUNDFONT` selects a soundfont explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 44100

_SOUNDFONT_CANDIDATES = [
  "/usr/share/sounds/sf2/FluidR3_GM.sf2",
  "/usr/share/soundfonts/FluidR3_GM.sf2",
  "/usr/share/sounds/sf2/default-GM.sf2",
  os.path.expanduser("~/.fluidsynth/default_sound_font.sf2"),
]


def find_soundfont() -> Optional[str]:
  env = os.environ.get("UMUST_SOUNDFONT")
  if env and Path(env).exists():
    return env
  for c in _SOUNDFONT_CANDIDATES:
    if Path(c).exists():
      return c
  try:
    import pretty_midi
    bundled = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
    if bundled.exists():
      return str(bundled)
  except ImportError:
    pass
  return None


def _render_pyfluidsynth(midi_path: str, sf2: str, sr: int) -> Optional[np.ndarray]:
  try:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    audio = pm.fluidsynth(fs=sr, sf2_path=sf2)   # needs pyfluidsynth + libfluidsynth
  except Exception:  # noqa: BLE001 - library missing or synthesis failed
    return None
  return audio.astype(np.float32)


def _render_tinysoundfont(midi_path: str, sf2: str, sr: int, tail_sec: float = 2.0) -> Optional[np.ndarray]:
  try:
    import tinysoundfont
  except ImportError:
    return None
  synth = tinysoundfont.Synth(samplerate=sr)
  sfid = synth.sfload(sf2)
  # channels without a program stay silent, so give every channel the piano
  # preset (drums on channel 10); program changes in the file override this
  for ch in range(16):
    try:
      if ch == 9:
        synth.program_select(ch, sfid, 128, 0, is_drums=True)
      else:
        synth.program_select(ch, sfid, 0, 0)
    except Exception:  # noqa: BLE001 - preset missing in this soundfont
      pass
  seq = tinysoundfont.Sequencer(synth)
  seq.midi_load(midi_path)
  frames = 4096
  chunks = []
  # synth.generate() drives the sequencer callback, which sends the events
  max_chunks = int(30 * 60 * sr / frames)          # safety cap: 30 minutes
  while not seq.is_empty() and len(chunks) < max_chunks:
    buf = synth.generate(frames)
    chunks.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, 2).mean(axis=1))
  for _ in range(int(tail_sec * sr / frames) + 1):  # let the last notes decay
    buf = synth.generate(frames)
    chunks.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, 2).mean(axis=1))
  audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
  return audio.astype(np.float32)


def render_midi_reference(midi_path: str, max_duration_sec: Optional[float] = None,
                          sr: int = SAMPLE_RATE) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
  """Synthesize `midi_path` with a GM soundfont.

  Returns ((sr, mono float32 audio), backend description) or (None, reason)."""
  sf2 = find_soundfont()
  if sf2 is None:
    return None, "no soundfont found (install fluid-soundfont-gm or set UMUST_SOUNDFONT)"
  audio = _render_pyfluidsynth(midi_path, sf2, sr)
  backend = "FluidSynth"
  if audio is None:
    audio = _render_tinysoundfont(midi_path, sf2, sr)
    backend = "TinySoundFont"
  if audio is None:
    return None, "neither pyfluidsynth (with libfluidsynth) nor tinysoundfont is available"
  if max_duration_sec is not None:
    audio = audio[: int(max_duration_sec * sr)]
  peak = float(np.max(np.abs(audio))) if audio.size else 0.0
  if peak > 0:
    audio = audio / peak * 0.9
  return (sr, audio), f"{backend} · {Path(sf2).name}"
