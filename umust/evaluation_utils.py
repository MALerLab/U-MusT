from collections import defaultdict
from typing import Union
import math
from math import log
import re

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import Levenshtein
from tqdm.auto import tqdm
from omegaconf import DictConfig

from .model_zoo import LatentScoreAMT
from .data_utils import MultimodalTokenDataset
from .vocab_utils import VQVocab
from .yourmt3plus.utils.metrics import compute_track_metrics
from .midi_utils.event2note import note_event2note

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


class Evaluator:
  def __init__(self):
    pass
  
  @classmethod
  def eval_omr_loader(cls, model: LatentScoreAMT, omr_loader):
    total_inf_out = []
    total_target_out = []
    for batch in tqdm(omr_loader):
      in_modal, in_mask, target_in, target_out, modal_idx, token_heights, in_pos, target_in_pos = batch['in_modal'], batch['in_mask'], batch['target_in'], batch['target_out'], batch['modal_idx'], batch['token_height'], batch['in_pos'], batch['target_in_pos']  
      inf_out = model.inference(in_modal, in_pos, modal_idx, in_mask=in_mask, temperature=0.1, max_length=model.out_vocab.max_seq_len['lmx'])
      total_inf_out.append(inf_out.cpu().to(torch.short))
      total_target_out.append(target_out.cpu().to(torch.short))

    gold_lmx, pred_lmx = [], []
    for inf_out, target_out in zip(total_inf_out, total_target_out):
      for idx in range(inf_out.shape[0]):
        gold = target_out[idx, :, 0] - model.out_vocab.idx_shifts['lmx']
        pred = inf_out[idx, :, 0] - model.out_vocab.idx_shifts['lmx']
        gold[gold<0] = 0
        pred[pred<0] = 0
        gold = model.out_vocab.vocabs['lmx'].decode(gold.cpu())
        pred = model.out_vocab.vocabs['lmx'].decode(pred.cpu())
        gold_lmx.append(gold)
        pred_lmx.append(pred)
    return calc_ser_metric(gold_lmx, pred_lmx)
  
  @classmethod
  def eval_amt_dataset(cls, model, dataset: MultimodalTokenDataset, decoder, target_dataset: str, batch_size=10, num_samples=10, max_len_sec=60, rank=0, world_size=1):
    target_path_pairs = [x[1] for x in dataset.path_pairs if x[0] == target_dataset]
    target_path_pairs = target_path_pairs[:num_samples]
    entire_out = defaultdict(list)
    if world_size > 1:
      target_path_pairs = target_path_pairs[rank::world_size]
    
    for path_pair in tqdm(target_path_pairs, desc="Evaluating AMT dataset"):
      dac_slices = dataset.make_piece_dac_slices(path_pair, dataset.midi_slice_len-0.5, max_len_sec=max_len_sec)
      num_batches = math.ceil(len(dac_slices) / batch_size)
      piece_output = []
      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        dac_batch = dac_slices[start_idx:end_idx]
        in_modal, in_pos, in_mask = pad_collate_fn(dac_batch)
        modal_idx = [model.in_vocab.vocab_keys.index('dac'), model.out_vocab.vocab_keys.index('midi')]
        modal_idx = torch.tensor(modal_idx).repeat(in_modal.shape[0], 1)
        output = model.inference(in_modal=in_modal, 
                                 in_pos=in_pos, 
                                 modal_idx=modal_idx, 
                                 in_mask=None, 
                                 token_heights=None, 
                                 temperature=0.1, 
                                 manual_seed=-1, 
                                 max_length=model.out_vocab.max_seq_len['midi'])
        piece_output.append(output.cpu()[...,0:1].to(torch.short))
      flattened_piece_output = [x for batch in piece_output for x in batch]
      notes = decoder.decode_piece_midi(flattened_piece_output, hop_len=dataset.midi_slice_len)
      ref_notes, _ = note_event2note(dataset.load_data(path_pair)['midi']['note_events'])
      out = compute_track_metrics(notes, ref_notes)[1] # does not use drum metrics
      for k, v in out.items():
        entire_out[k].append(v)
    for k, v in entire_out.items():
      entire_out[k] = sum(v) / len(v)
    return entire_out, len(target_path_pairs)



class LayerPeeper:
  def __init__(self, target, use_only_last=False, hook_fn=None, extract_fn=None):
    self.use_only_last = use_only_last
    self.output = None if use_only_last else []
    
    if hook_fn:
      self.handler = target.register_forward_hook(hook_fn(self))
    else:
      self.handler = target.register_forward_hook(self.store_intermediates(extract_fn))
  
  
  @property
  def last(self):
    if self.use_only_last:
      return self.output
    else:
      return self.output[-1]
  
  
  def store_intermediates(self, extract_fn):
    def hook_fn(module, input, output):
      if extract_fn:
        output = extract_fn(output)
      
      if self.use_only_last:
        self.output = output.clone().detach()
      else:
        self.output.append(output.clone().detach())
    
    return hook_fn
  
  
  def remove(self):
    self.handler.remove()


def use_attn_weights(self):
  def hook_fn(module, _input, output):
    _q, _k, *_ = _input
    q = _q.clone().detach()
    k = _k.clone().detach()
    
    with module.sdp_context_manager():
    # with torch.backends.cuda.sdp_kernel(**module.sdp_kwargs):
      scale_factor = 1 / math.sqrt(q.size(-1))
      attn_weight = q @ k.transpose(-2, -1) * scale_factor
      attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if self.use_only_last:
      self.output = attn_weight
    else:
      self.output.append(attn_weight)
  
  return hook_fn


def draw_attention_map(attn_weights: torch.Tensor):
  # attn_weights.shape: (n_timestep, num_heads, q_seq_len)
  n_heads = attn_weights.shape[1]
  
  fig = plt.figure(figsize=(8, 14), layout='tight', dpi=300.0)

  for i in range(n_heads):
    head = attn_weights[:, i, :].T.cpu()
    ax = fig.add_subplot(n_heads, 1, i+1)
    ax.matshow(head, interpolation='nearest')

  fig.canvas.draw()
  plt.close()

  return np.array(fig.canvas.renderer._renderer)



def levenshtein_distance_pure(a: list, b: list) -> int:
  len_a, len_b = len(a), len(b)

  distances = [j for j in range(len_b + 1)]
  for i in range(1, len_a + 1):
      prev_distances, distances = distances, [i] + [0] * len_b
      for j in range(1, len_b + 1):
        distances[j] = min(
          distances[j - 1] + 1, # insertion
          prev_distances[j] + 1, # deletion
          prev_distances[j - 1] + (a[i - 1] != b[j - 1]), # substitution or match
        )

  return distances[-1]


levenshtein_distance = Levenshtein.distance

tuplets_exceptions = {
  "tuplet:start",
  "tuplet:stop",
}
tuplets_exception_re = re.compile(r"^\d+in\d+$")


def calc_ser_metric(gold: list[str], pred: list[str]) -> dict:
  assert len(gold) == len(pred), "Gold and predicted data must have the same length"

  ser_errors, ser_total = 0, 0
  sert_errors, sert_total = 0, 0
  
  for gold_lmx, pred_lmx in zip(gold, pred):
    gold_lmx = gold_lmx.rstrip("\r\n").split()
    pred_lmx = pred_lmx.rstrip("\r\n").split()

    ser_errors += levenshtein_distance(gold_lmx, pred_lmx)
    ser_total += len(gold_lmx)

    gold_tuplets = [
      x 
      for x in gold_lmx 
      if x not in tuplets_exceptions and not tuplets_exception_re.match(x)
    ]
    pred_tuplets = [
      x 
      for x in pred_lmx 
      if x not in tuplets_exceptions and not tuplets_exception_re.match(x)
    ]
    
    sert_errors += levenshtein_distance(gold_tuplets, pred_tuplets)
    sert_total += len(gold_tuplets)

  assert ser_total > 0, "Gold data cannot be empty"
  
  return {
    "SER": 100 * ser_errors / ser_total, 
    "SERnotuplets": 100 * sert_errors / sert_total
  }