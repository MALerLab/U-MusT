from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import time
import os
import math

import cv2


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from umust.utils import *
from umust.data_decode_utils import TensorDecoder
from umust.yourmt3plus.utils.metrics import compute_track_metrics
from umust.midi_utils.event2note import note_event2note

def wandb_style_config_to_omega_config(wandb_conf):
  # remove wandb related config
  for wandb_key in ["wandb_version", "_wandb"]:
    if wandb_key in wandb_conf:
      del wandb_conf[wandb_key] # wandb-related config should not be overrided! 

  # remove nonnecessary fields such as desc and value
  for key in wandb_conf:
    if 'desc' in wandb_conf[key]:
      del wandb_conf[key]['desc']
    if 'value' in wandb_conf[key]:
      wandb_conf[key] = wandb_conf[key]['value']
  return wandb_conf
def pad_collate_fn(batch):
  
  max_len = max([x[0].shape[0] for x in batch])
  padded_batch = torch.zeros(len(batch), max_len, batch[0][0].shape[1], dtype=torch.long)
  in_pos_padded_batch = torch.zeros(len(batch), max_len, 2, dtype=torch.long)
  in_mask_padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
  for i, x in enumerate(batch):
    padded_batch[i, :x[0].shape[0]] = x[0]
    in_pos_padded_batch[i, :x[1].shape[0]] = x[1]
    in_mask_padded_batch[i, :x[0].shape[0]] = 1
  return padded_batch, in_pos_padded_batch, in_mask_padded_batch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import argparse
parser = argparse.ArgumentParser(description='AMT evaluation (note onset F1) on MAESTRO / MusicNet / SLakh test sets')
parser.add_argument('--run_path', type=Path, required=True, help='training run directory containing files/config.yaml and files/checkpoints/')
parser.add_argument('--target_dataset', type=str, default='musicnet', choices=['musicnet', 'maestro', 'slakh'])
parser.add_argument('--data_dir', type=str, default='dataset/', help='root directory of the preprocessed datasets')
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

target_dataset = args.target_dataset
run_path = args.run_path
out_dir = run_path / "files" / "out"

batch_size = args.batch_size

dac_len_sec = 60



config_path = run_path / "files" / "config.yaml"
wandb_config = OmegaConf.load(config_path)
try:
  config = wandb_style_config_to_omega_config(wandb_config)
except Exception as e:
  print(e)
  config = wandb_config
print(config)
# config.data.lmx_vocab_path = 'vocab/lmx_vocab_singletoken_asap.txt'
config.data.data_dir = args.data_dir
# config.data.data_path[0] = ['lsyt_piano', 'latent_score_dataset_tokens/', ['dac', 'pt'], 0.02, [50000, 70000]]
config.data.preload_data = False
ckpt_paths = (run_path / 'files' / "checkpoints").glob("*.pt")
ckpt_paths = [x for x in ckpt_paths if 'iter' in x.stem]
last_ckpt_path = Path(max(ckpt_paths, key=lambda p: int(p.stem.split('_')[0][4:])))
# last_ckpt_path  = list(ckpt_paths)[0]

vq_model, vq_emb = get_vq_model(config)
dac_model, dac_emb = get_dac_model(config)
dataset = get_dataset(config)
model = get_model(config, vq_emb, dac_emb, dataset)

model.load_state_dict(torch.load(last_ckpt_path, map_location="cpu")["model_state_dict"])
model.eval()
model =model.cuda()
slice_len = config.data.midi_slice_len

decoder = TensorDecoder(config, model.in_vocab, model.out_vocab, out_dir, device='cuda:0')
train_set, val_set, test_set = dataset.get_datasets()


target_path_pairs = [x[1] for x in test_set.path_pairs if x[0] == target_dataset]

entire_out = []

for path_pair in target_path_pairs:
  dac_slices = test_set.make_piece_dac_slices(path_pair, slice_len)
  num_batches = math.ceil(len(dac_slices) / batch_size)
  piece_output = []
  for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    dac_batch = dac_slices[start_idx:end_idx]
    in_modal, in_pos, in_mask = pad_collate_fn(dac_batch)
    modal_idx = [model.in_vocab.vocab_keys.index('dac'), model.out_vocab.vocab_keys.index('midi')]
    modal_idx = torch.tensor(modal_idx).repeat(in_modal.shape[0], 1)
    output = model.inference(in_modal=in_modal, in_pos=in_pos, modal_idx=modal_idx, in_mask=None, token_heights=None, sampling_method='top_p', threshold=0.9, temperature=0.1, manual_seed=-1)
    piece_output.append(output.cpu()[...,0:1].to(torch.short))
  flattened_piece_output = [x for batch in piece_output for x in batch]
  notes, (out_path, out_path_mp3) = decoder.decode_piece_midi(flattened_piece_output, hop_len=slice_len, filename=path_pair['midi'].stem)
  ref_notes, _ = note_event2note(val_set.load_data(path_pair)['midi']['note_events'])
  out = compute_track_metrics(notes, ref_notes)
  print(path_pair['midi'].stem, out[1])
  entire_out.append(out[1])

print(entire_out)