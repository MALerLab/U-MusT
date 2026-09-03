from pathlib import Path
import json
import torch
from tqdm.auto import tqdm

dataset_dir = Path('dataset/olimpic')  # root of the OLiMPiC dataset
output_dir = Path('dataset_pair_paths/')

pt_fns = sorted(list(dataset_dir.rglob('*.pt')))

pairs = []
for pt_fn in tqdm(pt_fns):
  target_lmx_fn = pt_fn.parent.parent.parent.parent / (pt_fn.stem.replace('-crop_resize', '') + '.lmx')
  if target_lmx_fn.exists():
    pt_data = torch.load(pt_fn, map_location='cpu')
    pairs.append({
      'lmx': str(target_lmx_fn.relative_to(dataset_dir)),
      'pt': str(pt_fn.relative_to(dataset_dir)),
      'pt_len': pt_data.shape[-2] * pt_data.shape[-3]
    })

split_txt_names = {"valid": "samples.dev.txt", "train": "samples.train.txt", "test": "samples.test.txt"}
subdata_names = ["olimpic-1.0-synthetic", "olimpic-1.0-scanned"]

split_dict = {}
for split_name, split_txt_name in split_txt_names.items():
  split_dict[split_name] = []
  for subdata_name in subdata_names:
    txt_fn = dataset_dir / subdata_name / split_txt_name
    if not txt_fn.exists():
      continue
    with open(txt_fn, 'r') as f:
      lines = f.readlines()
    split_dict[split_name].extend([x.strip() for x in lines])

dict_by_piece = {}
for k, v in split_dict.items():
  for x in v:
    dict_by_piece[x] = k

train_pairs = [x for x in pairs if dict_by_piece['/'.join(x['lmx'].split('/')[1:]).replace('.lmx', '')] == 'train']
valid_pairs = [x for x in pairs if dict_by_piece['/'.join(x['lmx'].split('/')[1:]).replace('.lmx', '')] == 'valid']
test_pairs = [x for x in pairs if dict_by_piece['/'.join(x['lmx'].split('/')[1:]).replace('.lmx', '')] == 'test']

data = {
  'train': train_pairs,
  'valid': valid_pairs,
  'test': test_pairs
}

with open(output_dir / 'olimpic-lmx.json', 'w') as f:
  json.dump(data, f, indent=2)