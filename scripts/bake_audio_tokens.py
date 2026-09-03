"""Bake shift-augmented DAC audio tokens (paper Sec. III-A3, Appendix B).

Audio is resampled to 44.1 kHz mono and cut into fixed-length segments
(60 s with a 50 s hop by default). Each segment is encoded 9 times under
temporal shifts of -20..+20 samples at 5-sample intervals (one 5-sample
shift is about 0.113 ms), stacked in the batch dimension of a single
DACFile:

  <audio_parent>/mono_augmented/<dac_model>_<n_codebook>/<stem>_<start_sec:04d>.dac
      codes shape: (9, n_codebook, T)   [variant 4 = unshifted]

This matches the layout of the released training tokens
(e.g. maestro/audio_tokens/.../mono_augmented/unidac4_4/*_0000.dac).

Example:
  python3 scripts/bake_audio_tokens.py dataset/maestro --pattern "*.wav"
"""
import argparse
import sys
from pathlib import Path

import torch
from audiotools import AudioSignal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umust.dac_utils import LSDAC

SAMPLE_RATE = 44100
SHIFTS = list(range(-20, 21, 5))  # 9 variants; index 4 is the unshifted signal
PAD = max(abs(s) for s in SHIFTS)


def main():
  parser = argparse.ArgumentParser(description='Bake shift-augmented DAC audio tokens.')
  parser.add_argument('audio_dir', type=Path, help='root directory searched recursively for audio files')
  parser.add_argument('--pattern', default='*.wav', help='glob pattern for audio files')
  parser.add_argument('--dac_model', default='unidac4', help='audio tokenizer name')
  parser.add_argument('--dac_model_dir', default='dac_models', help='directory holding the tokenizer checkpoints')
  parser.add_argument('--n_codebook', type=int, default=4)
  parser.add_argument('--segment_sec', type=int, default=60)
  parser.add_argument('--hop_sec', type=int, default=50)
  parser.add_argument('--min_tail_sec', type=float, default=1.0, help='skip a trailing segment shorter than this')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--device', default='cuda')
  args = parser.parse_args()

  model = LSDAC.load(Path(args.dac_model_dir) / args.dac_model / 'weights.pth')
  model = model.to(args.device).eval()
  torch.set_grad_enabled(False)

  out_dirname = f'{args.dac_model}_{args.n_codebook}'
  seg_len = args.segment_sec * SAMPLE_RATE
  hop_len = args.hop_sec * SAMPLE_RATE

  audio_path_list = sorted(args.audio_dir.rglob(args.pattern))
  print(f'{len(audio_path_list)} audio files to bake')

  for audio_path in tqdm(audio_path_list):
    out_dir = audio_path.parent / 'mono_augmented' / out_dirname
    signal = AudioSignal(str(audio_path))
    if signal.sample_rate != SAMPLE_RATE:
      signal = signal.resample(SAMPLE_RATE)
    signal = signal.to_mono()
    data = signal.audio_data.reshape(-1)  # (T,)
    padded = torch.nn.functional.pad(data, (PAD, PAD))

    n_samples = data.shape[-1]
    for start in range(0, n_samples, hop_len):
      if n_samples - start < args.min_tail_sec * SAMPLE_RATE:
        break
      save_path = out_dir / f'{audio_path.stem}_{start // SAMPLE_RATE:04d}.dac'
      if args.skip_existing and save_path.exists():
        continue
      length = min(seg_len, n_samples - start)
      variants = torch.stack([
        padded[PAD + start + shift: PAD + start + shift + length]
        for shift in SHIFTS
      ]).unsqueeze(1)  # (9, 1, length)
      seg_signal = AudioSignal(variants, SAMPLE_RATE)
      dac_file = model.compress(seg_signal, win_duration=args.segment_sec + 2)
      dac_file.codes = dac_file.codes.cpu()
      save_path.parent.mkdir(parents=True, exist_ok=True)
      dac_file.save(save_path)


if __name__ == '__main__':
  main()
