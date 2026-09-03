"""Build YTSV split manifests (lsyt.json / lsyt_piano.json) from tokenized data.

Scans a tokenized YTSV tree (image tokens `.pt` + audio tokens `.dac`,
produced by the YTSV pipeline / scripts/bake_*_tokens.py), pairs them per
segment, and writes a train/valid/test manifest compatible with
MultimodalTokenDatasetMaker.

The top-level group directories encode instrumentation as
`<melody_staves>-<piano_staves>` (e.g. `0-2` = piano solo, `2-0` = two
melody instruments). Use --groups to restrict, e.g. the piano subset:

  python3 scripts/make_youtube_split_json.py dataset/latent_score_dataset_tokens \
    --groups "0-*" --output dataset_pair_paths/lsyt_piano.json

  python3 scripts/make_youtube_split_json.py dataset/latent_score_dataset_tokens \
    --output dataset_pair_paths/lsyt.json
"""
import argparse
import fnmatch
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from dac import DACFile
from tqdm.auto import tqdm

TOKEN_DIR = None  # set in main; module-level for worker processes


def build_pair(args):
  seg_id, dac_fn, pt_fns = args
  try:
    pt_datas = [torch.load(t, map_location='cpu', weights_only=False) for t in pt_fns]
    return {
      'dac': str(dac_fn.relative_to(TOKEN_DIR)),
      'pt': [str(x.relative_to(TOKEN_DIR)) for x in pt_fns],
      'yt_id': seg_id.split(':')[0],
      'dac_len': DACFile.load(dac_fn).codes.shape[-1],
      'pt_len': sum([t.shape[-2] for t in pt_datas]) * max([t.shape[-3] for t in pt_datas]),
    }
  except Exception as e:
    print(f'Error for {seg_id}: {e}')
    return None


def init_worker(token_dir):
  global TOKEN_DIR
  TOKEN_DIR = token_dir


def main():
  global TOKEN_DIR
  parser = argparse.ArgumentParser(description='Build YTSV split manifests from tokenized data.')
  parser.add_argument('token_dir', type=Path, help='root of the tokenized YTSV dataset')
  parser.add_argument('--output', type=Path, default=Path('dataset_pair_paths/lsyt.json'))
  parser.add_argument('--groups', default=None, help="glob over top-level group dirs, e.g. '0-*' for piano-only")
  parser.add_argument('--test_ids_csv', type=Path, default=Path('dataset_pair_paths/test_included_yt_ids.csv'),
                      help='videos that must go to the test split (overlap with AMT test sets)')
  parser.add_argument('--workers', type=int, default=16)
  args = parser.parse_args()
  TOKEN_DIR = args.token_dir

  groups = sorted(d for d in args.token_dir.iterdir() if d.is_dir())
  if args.groups is not None:
    groups = [d for d in groups if fnmatch.fnmatch(d.name, args.groups)]
  print(f'groups: {[d.name for d in groups]}')

  dac_fns, pt_fns = [], []
  for g in groups:
    dac_fns += list(g.rglob('*.dac'))
    pt_fns += list(g.rglob('*.pt'))
  dac_fns, pt_fns = sorted(dac_fns), sorted(pt_fns)
  print(f'Len dac: {len(dac_fns)}, len pt: {len(pt_fns)}')

  dac_fn_by_id = {':'.join(x.stem.split(':')[:2]): x for x in dac_fns}
  pt_fn_by_id = defaultdict(list)
  for pt_fn in pt_fns:
    pt_fn_by_id[':'.join(pt_fn.stem.split(':')[:2])].append(pt_fn)

  jobs = [(k, v, pt_fn_by_id[k]) for k, v in dac_fn_by_id.items() if len(pt_fn_by_id[k]) > 0]
  pairs = []
  with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.token_dir,)) as ex:
    for pair in tqdm(ex.map(build_pair, jobs, chunksize=64), total=len(jobs)):
      if pair is not None:
        pairs.append(pair)

  test_included_yt_ids = pd.read_csv(args.test_ids_csv)['YT Id'].tolist()
  yt_ids = sorted(list(set([x['yt_id'] for x in pairs])))
  non_test_yt_ids = [x for x in yt_ids if x not in test_included_yt_ids]
  included_yt_ids = [x for x in yt_ids if x in test_included_yt_ids]
  print(len(non_test_yt_ids), len(included_yt_ids))

  random.seed(0)
  random.shuffle(non_test_yt_ids)

  ratio_train = 0.85
  ratio_valid = 0.07

  train_yt_ids = non_test_yt_ids[:int(len(yt_ids) * ratio_train)]
  valid_yt_ids = non_test_yt_ids[int(len(yt_ids) * ratio_train):int(len(yt_ids) * (ratio_train + ratio_valid))]
  test_yt_ids = non_test_yt_ids[int(len(yt_ids) * (ratio_train + ratio_valid)):] + included_yt_ids
  assert len(train_yt_ids) + len(valid_yt_ids) + len(test_yt_ids) == len(yt_ids)

  id_to_split = {}
  for yt_id in test_yt_ids:
    id_to_split[yt_id] = 'test'
  for yt_id in valid_yt_ids:
    id_to_split[yt_id] = 'valid'
  for yt_id in train_yt_ids:
    id_to_split[yt_id] = 'train'

  data = {split: [x for x in pairs if id_to_split[x['yt_id']] == split]
          for split in ('train', 'valid', 'test')}
  print({k: len(v) for k, v in data.items()})

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(data, f, indent=2)
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
