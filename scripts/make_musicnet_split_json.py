from pathlib import Path
import numpy as np
from dac import DACFile
from collections import defaultdict
import json
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
  dataset_dir = Path('dataset/musicnet')  # root of the MusicNet dataset
  output_dir = Path('dataset_pair_paths/')
  assert dataset_dir.exists()




  dac_fns = sorted(list(dataset_dir.rglob('*.dac')))
  dac_fns = [fn for fn in dac_fns if '_augmented' in str(fn)]

  dac_fn_by_file_idx = defaultdict(list)
  for dac_fn in dac_fns:
    file_idx = dac_fn.stem.split('_')[0]
    dac_fn_by_file_idx[file_idx].append(dac_fn)

  pairs = []
  for k, v in dac_fn_by_file_idx.items():
    npy_fn = dataset_dir / 'musicnet_em' / f"{k}_note_events.npy"
    if npy_fn.exists():
      pairs.append((v, npy_fn))
      
  test_pairs, train_pairs = split_by_ids(pairs, test_ids)
  test_pairs_str = [convert_pair_to_str(pair, dataset_dir) for pair in test_pairs]
  train_pairs_str = [convert_pair_to_str(pair, dataset_dir) for pair in train_pairs]
  
  import random
  random.seed(0)
  random.shuffle(train_pairs_str)
  train_pairs_str = train_pairs_str[:-10]
  valid_pairs_str = train_pairs_str[-10:]
  

  with open(output_dir / 'musicnet.json', 'w') as f:
    json.dump({
      'test': test_pairs_str,
      'train': train_pairs_str,
      'valid': valid_pairs_str
    }, f, indent=2)


if __name__ == '__main__':
  main()