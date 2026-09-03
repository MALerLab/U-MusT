from pathlib import Path
import numpy as np
from dac import DACFile
from collections import defaultdict
import json
import pandas as pd
from tqdm import tqdm
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
  dataset_dir = Path("dataset/slakh")  # root of the SLakh dataset
  output_dir = Path('dataset_pair_paths/')
  assert dataset_dir.exists()
  
  npy_fns = list(dataset_dir.rglob('*_note_events.npy'))
  
  pairs = []
  for npy_fn in tqdm(npy_fns):
    dac_dir = npy_fn.parent / 'audio_tokens' / 'mix'
    dac_fns = sorted(list(dac_dir.rglob("*.dac")))
    dac_fns = [fn for fn in dac_fns if '_augmented' in str(fn)]
    if len(dac_fns) == 0:
      print(f"No dac files found for {npy_fn}")
      continue
    pairs.append((dac_fns, npy_fn))
    
  

  train_pairs = [pair for pair in pairs if '/train/' in str(pair[1])]
  valid_pairs = [pair for pair in pairs if '/validation/' in str(pair[1])]
  test_pairs = [pair for pair in pairs if '/test/' in str(pair[1])]

  assert len(train_pairs) + len(valid_pairs) + len(test_pairs) == len(pairs)
  
  train_pairs = [convert_pair_to_str(pair, dataset_dir) for pair in train_pairs]
  valid_pairs = [convert_pair_to_str(pair, dataset_dir) for pair in valid_pairs]
  test_pairs = [convert_pair_to_str(pair, dataset_dir) for pair in test_pairs]
  
  with open(output_dir/"slakh.json", "w") as f:
    json.dump({'train': train_pairs, 'valid': valid_pairs, 'test': test_pairs}, f, indent=2)
if __name__ == '__main__':
  main()
