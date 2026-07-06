from pathlib import Path
import numpy as np
from dac import DACFile
from collections import defaultdict
import json
import pandas as pd

test_ids = [2191, 2628, 2106, 2298, 1819, 2416]

def convert_pair_to_str(pair, base_dir):
  return {
    'dac': [str(p.relative_to(base_dir)) for p in pair[0]],
    'npy': str(pair[1].relative_to(base_dir))
  }

def split_by_ids(pairs, test_ids):
  test_pairs = [pair for pair in pairs if int(pair[1].stem.split('_')[0]) in test_ids]
  train_pairs = [pair for pair in pairs if int(pair[1].stem.split('_')[0]) not in test_ids]
  return test_pairs, train_pairs


def main():
  dataset_dir = Path('dataset/maestro/')  # root of the MAESTRO dataset
  output_dir = Path('dataset_pair_paths/')
  assert dataset_dir.exists()
  
  npy_fns = list(dataset_dir.rglob('*_note_events.npy'))
  
  dac_fns = sorted(list(dataset_dir.rglob('*.dac')))
  dac_fns = [fn for fn in dac_fns if '_augmented' in str(fn)]
  dac_fn_by_file_idx = defaultdict(list)
  for dac_fn in dac_fns:
    file_idx = '_'.join(dac_fn.stem.split('_')[:-1])
    dac_fn_by_file_idx[file_idx].append(dac_fn)
    
  pairs = []
  for npy_fn in npy_fns:
    dac_fns = dac_fn_by_file_idx[npy_fn.stem.replace('_note_events', '')]
    pairs.append((dac_fns, npy_fn))
  
  df = pd.read_csv(dataset_dir/"maestro-v3.0.0.csv")
  train_fns = [Path(fn).stem for fn in df[df['split']=="train"]['audio_filename'].tolist()]
  valid_fns = [Path(fn).stem for fn in df[df['split']=="validation"]['audio_filename'].tolist()]
  test_fns = [Path(fn).stem for fn in df[df['split']=="test"]['audio_filename'].tolist()]
  
  train_pairs = [pair for pair in pairs if pair[1].stem.replace('_note_events', '') in train_fns]
  valid_pairs = [pair for pair in pairs if pair[1].stem.replace('_note_events', '') in valid_fns]
  test_pairs = [pair for pair in pairs if pair[1].stem.replace('_note_events', '') in test_fns]
  
  train_pairs = [convert_pair_to_str(pair, dataset_dir) for pair in train_pairs]
  valid_pairs = [convert_pair_to_str(pair, dataset_dir) for pair in valid_pairs]
  test_pairs = [convert_pair_to_str(pair, dataset_dir) for pair in test_pairs]
  
  with open(output_dir/"maestro.json", "w") as f:
    json.dump({'train': train_pairs, 'valid': valid_pairs, 'test': test_pairs}, f, indent=2)
if __name__ == '__main__':
  main()
