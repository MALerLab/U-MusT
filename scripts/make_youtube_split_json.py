from collections import defaultdict
from pathlib import Path
import random
import pandas as pd
from dac import DACFile
import torch
from tqdm.auto import tqdm

token_dir = Path('dataset/ytsv_tokens')  # root of the tokenized YTSV dataset
output_dir = Path('dataset_pair_paths/')
test_yt_ids = pd.read_csv("dataset_pair_paths/test_included_yt_ids.csv")['YT Id'].tolist()

dac_fns = sorted(list(token_dir.rglob('*.dac')))
pt_fns = sorted(list(token_dir.rglob('*.pt')))

print(f"Len dac: {len(dac_fns)}, len pt: {len(pt_fns)}")
print(f"Dac: {dac_fns[0]}, pt: {pt_fns[0]}")
print(f"Dac: {dac_fns[-1]}, pt: {pt_fns[-1]}")

dac_fn_by_id = {':'.join(x.stem.split(':')[:2]): x for x in dac_fns}
pt_fn_by_id = defaultdict(list) 
for pt_fn in pt_fns:
  pt_fn_by_id[':'.join(pt_fn.stem.split(':')[:2])].append(pt_fn)

pairs = []
for k, v in tqdm(dac_fn_by_id.items()):
  corresp_pt = pt_fn_by_id[k]
  try:
    if len(corresp_pt) > 0:
      pt_datas = [torch.load(t, map_location='cpu') for t in corresp_pt]
      pairs.append({
        'dac': str(v.relative_to(token_dir)),
        'pt': [str(x.relative_to(token_dir)) for x in corresp_pt],
        'yt_id': k.split(':')[0],
        'dac_len': DACFile.load(v).codes.shape[-1],
        'pt_len': sum([t.shape[-2] for t in pt_datas]) * max([t.shape[-3] for t in pt_datas])
      })
  except Exception as e:
    print(f"Error for {k}: {e}")
    print(f"Corresponding pt files: {corresp_pt}")
    continue
    


yt_ids = sorted(list(set([x['yt_id'] for x in pairs])))
non_test_yt_ids = [x for x in yt_ids if x not in test_yt_ids]
included_yt_ids = [x for x in yt_ids if x in test_yt_ids]
print(len(non_test_yt_ids), len(included_yt_ids))

random.seed(0)
random.shuffle(non_test_yt_ids)

ratio_train = 0.85
ratio_valid = 0.07
ratio_test = 0.08

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

train_pairs = [x for x in pairs if id_to_split[x['yt_id']] == 'train']
valid_pairs = [x for x in pairs if id_to_split[x['yt_id']] == 'valid']
test_pairs = [x for x in pairs if id_to_split[x['yt_id']] == 'test']

data = {
  'train': train_pairs,
  'valid': valid_pairs,
  'test': test_pairs
}

import json
with open(output_dir / 'lsyt.json', 'w') as f:
  json.dump(data, f, indent=2)
