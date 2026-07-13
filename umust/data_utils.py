from pathlib import Path
import csv
import gzip
from typing import Union, List, Optional, Iterator
from tqdm import tqdm
import json

from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

import mido
import numpy as np
import random
import math
from sklearn.cluster import KMeans
from dac import DACFile

from .vocab_utils import VQVocab, LMXVocab, RVQVocab, TokenIdxHandler
from .constants import *
from .midi_utils.tokenizer import NoteEventTokenizer
from .midi_utils.note2event import slice_note_events_and_ties
from .lmx_utils.lmx_slice import slice_by_measure, get_measure_boundary_from_lmx


def audio_token_collate_fn(batch, pad_idx, rvq=False):
  audio_lengths = [item[0].shape[-1] for item in batch]
  max_audio_length = max(audio_lengths)
  if rvq:
    max_tok_length = max([item[1].shape[-2] for item in batch])
  else:
    max_tok_length = max([item[1].shape[-1] for item in batch])
  audios = []
  toks = []
  for item in batch:
    audios.append(F.pad(item[0], (0, max_audio_length - item[0].shape[-1]), mode='constant', value=0))
    if rvq:
      toks.append(F.pad(item[1].permute(0,2,1), (0, max_tok_length - item[1].shape[-2]), mode='constant', value=pad_idx).permute(0,2,1))
    else:
      toks.append(F.pad(item[1], (0, max_tok_length - item[1].shape[-1]), mode='constant', value=pad_idx))
  return torch.cat(audios, dim=0), torch.tensor(audio_lengths), torch.cat(toks, dim=0).long()

def pianoroll_collate_fn(batch):
  audios = [item[0] for item in batch]
  pianorolls = [item[1] for item in batch]
  audio_lengths = [audio.shape[-1] for audio in audios]
  max_audio_length = max(audio_lengths)
  padded_audios = []
  padded_pianorolls = []
  max_pianoroll_length = max([pianoroll.shape[-1] for pianoroll in pianorolls])
  for audio, pianoroll in zip(audios, pianorolls):
    padded_audios.append(F.pad(audio, (0, max_audio_length - audio.shape[-1]), mode='constant', value=0))
    padded_pianorolls.append(F.pad(pianoroll, (0, max_pianoroll_length - pianoroll.shape[-1]), mode='constant', value=0))
  return (
    torch.cat(padded_audios, dim=0),
    torch.tensor(audio_lengths),
    torch.stack(padded_pianorolls),
  )

def lmx_vq_collate_fn(batch, lmx_pad_idx, vq_pad_idx, rvq=False):
  """
  Collates a batch of data for Linearized MusicXML to and vector quantization (VQ) tasks.

  Args:
    batch (list): A list of tuples where each tuple contains two tensors. The first tensor is for LMX and the second tensor is for VQ.
    lmx_pad_idx (int): Padding index for LMX tokens.
    vq_pad_idx (int): Padding index for VQ tokens.
    rvq (bool, optional): If True, the VQ tensors are permuted and padded along the second last dimension. Defaults to False.

  Returns:
    tuple: A tuple containing:
      - lmx_tokens (torch.Tensor): Padded and concatenated LMX tokens.
      - lmx_masks (torch.Tensor): Mask tensor indicating the valid lengths of LMX tokens.
      - vq_tokens (torch.Tensor): Padded and concatenated VQ tokens.
  """
  lmx_lengths = [item[0].shape[-1] for item in batch]
  max_lmx_length = max(lmx_lengths)
  if rvq:
    max_vq_length = max([item[1].shape[-2] for item in batch])
  else:
    max_vq_length = max([item[1].shape[-1] for item in batch])
  lmx_tokens = []
  vq_tokens = []
  for item in batch:
    lmx_tokens.append(F.pad(item[0], (0, max_lmx_length - item[0].shape[-1]), mode='constant', value=lmx_pad_idx))
    if rvq:
      vq_tokens.append(F.pad(item[1].permute(0,2,1), (0, max_vq_length - item[1].shape[-2]), mode='constant', value=vq_pad_idx).permute(0,2,1))
    else:
      vq_tokens.append(F.pad(item[1], (0, max_vq_length - item[1].shape[-1]), mode='constant', value=vq_pad_idx))
  lmx_tokens = torch.cat(lmx_tokens, dim=0).long()
  lmx_masks = torch.arange(lmx_tokens.shape[-1]).expand(len(batch), lmx_tokens.shape[-1]) < torch.tensor(lmx_lengths).unsqueeze(1)
  return lmx_tokens, lmx_masks, torch.cat(vq_tokens, dim=0).long()

def multimodal_collate_fn(batch, target_n_codebook, max_len=None, max_out_len=None):
  in_modal, target_in, target_out, modal_types, token_heights, in_pos, target_in_pos = zip(*batch)
  max_codebook = max([x.shape[1] for x in in_modal])
  if max_len is None:
    max_len = max([x.shape[0] for x in in_modal])
  if max_out_len is None:
    max_out_len = max([x.shape[0] for x in target_in])
  max_out_codebook = target_n_codebook
  in_modal = [torch.nn.functional.pad(x, (0, max_codebook-x.shape[1], 0, max_len-x.shape[0])) for x in in_modal]
  target_in = [torch.nn.functional.pad(x, (0, max_out_codebook-x.shape[1], 0, max_out_len-x.shape[0])) for x in target_in]
  target_out = [torch.nn.functional.pad(x, (0, max_out_codebook-x.shape[1], 0, max_out_len-x.shape[0])) for x in target_out]
  in_pos = [torch.nn.functional.pad(x, (0, 0, 0, max_len-x.shape[0])) for x in in_pos]
  target_in_pos = [torch.nn.functional.pad(x, (0, 0, 0, max_out_len-x.shape[0])) for x in target_in_pos]

  in_modal = torch.stack(in_modal)
  target_in = torch.stack(target_in)
  target_out = torch.stack(target_out)
  modal_types = torch.stack(modal_types)
  token_heights = torch.stack(token_heights)
  in_pos = torch.stack(in_pos)
  target_in_pos = torch.stack(target_in_pos) 

  in_mask = torch.ones((len(in_modal), max_len), dtype=torch.bool)
  in_mask[(in_modal[..., 0] == 0)] = 0
  return {'in_modal': in_modal, 'in_mask': in_mask, 'target_in': target_in, 'target_out': target_out, 'modal_idx': modal_types, 'token_height': token_heights, 'in_pos': in_pos, 'target_in_pos': target_in_pos}


def cal_weight_by_iter(max_weight, cur_iter, min_iter, max_iter):
  if cur_iter < min_iter:
    return 0.0
  elif cur_iter >= max_iter:
    return max_weight
  if min_iter == max_iter:
    return max_weight
  return max_weight * min(1, (cur_iter - min_iter) / (max_iter - min_iter))

class DistributedTwoDimKBucketSampler(torch.utils.data.distributed.DistributedSampler): # TODO Need to check
  def __init__(self, num_replicas, rank, lengths, n_buckets=6, shuffle=True, batch_size=32, drop_last=False):
    super().__init__(lengths, num_replicas, rank)
    
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.drop_last = drop_last
    
    assert isinstance(n_buckets, int)

    MAX_ITER = 200
    N_INIT = 10
    
    kmeans = KMeans(
      n_clusters=n_buckets, 
      max_iter=MAX_ITER, 
      n_init=N_INIT, 
      random_state=0
    ).fit(
      np.array( [ (l, w) for l, w in lengths ] )
    )
    
    clusters = [ [] for _ in range(kmeans.n_clusters) ]
    for d_idx, label in enumerate(kmeans.labels_):
      clusters[label].append( d_idx )
    
    self.buckets = dict()
    for i, bucket in enumerate(clusters):
      assert len(bucket) > 0 # should not be empty
      self.buckets[i] = torch.tensor(bucket, dtype=torch.int, device='cpu')
        
    if self.shuffle == True:
      for bucket_size in self.buckets.keys():
        self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]
            
    self.batches = []
    for bucket in self.buckets.values():
      curr_bucket = torch.split(bucket, self.batch_size)
      if len(curr_bucket) > 1 and self.drop_last == True:
        if len(curr_bucket[-1]) < len(curr_bucket[-2]):
          curr_bucket = curr_bucket[:-1]
      self.batches += curr_bucket
        
    self.length = len(self.batches)
    
    if self.shuffle == True:
      random.shuffle(self.batches)
  
  def __iter__(self):
    indices = list(range(self.length))
    if self.shuffle:
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(indices), generator=g).tolist()

    # subsample
    indices = indices[self.rank:self.length:self.num_replicas]
    
    for idx in indices:
        yield self.batches[idx].tolist()
  
  def __len__(self):
    return self.length
  
class TwoDimKBucketSampler(torch.utils.data.Sampler):
  def __init__(self, lengths, n_buckets=6, shuffle=True, batch_size=32, drop_last=False):
    """
    lengths: tuple of lists, [ (audio_length or lmx_length, image_width or lmx_width) ]
    buckets: should be int
    """
    super().__init__()
    
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.drop_last = drop_last
    
    assert isinstance(n_buckets, int), "ImageAudio Bucket Smapler's `n_bucket` arg should be int"
    
    MAX_ITER = 200
    N_INIT = 10
    
    kmeans = KMeans(
      n_clusters=n_buckets, 
      max_iter=MAX_ITER, 
      n_init=N_INIT, 
      random_state=0
    ).fit(
      np.array( [ (l, w) for l, w in lengths ] )
    )
    
    clusters = [ [] for _ in range(kmeans.n_clusters) ]
    for d_idx, label in enumerate(kmeans.labels_):
      clusters[label].append( d_idx )
    
    self.buckets = dict()
    for i, bucket in enumerate(clusters):
      assert len(bucket) > 0 # should not be empty
      self.buckets[i] = torch.tensor(bucket, dtype=torch.int, device='cpu')
        
    if self.shuffle == True:
      for bucket_size in self.buckets.keys():
        self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]
            
    self.batches = []
    for bucket in self.buckets.values():
      curr_bucket = torch.split(bucket, self.batch_size)
      if len(curr_bucket) > 1 and self.drop_last == True:
        if len(curr_bucket[-1]) < len(curr_bucket[-2]):
          curr_bucket = curr_bucket[:-1]
      self.batches += curr_bucket
        
    self.length = len(self.batches)
    
    if self.shuffle == True:
      random.shuffle(self.batches)
  
  def __iter__(self):
    for i in range(self.length):
      yield self.batches[i].tolist()
  
  def __len__(self):
    return self.length

class OneDimBucketSampler:
  def __init__(self, lengths, n_buckets=10, shuffle=True, batch_size=32, drop_last=False):
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
    
    assert isinstance(n_buckets, int), "AudioBucketSampler's `n_bucket` arg should be int"
    
    # Calculate bucket boundaries
    min_length = min(lengths)
    max_length = max(lengths)
    bucket_size = (max_length - min_length) // n_buckets
    
    # Create buckets
    self.buckets = {}
    for i in range(n_buckets):
      lower_bound = min_length + i * bucket_size
      upper_bound = lower_bound + bucket_size if i < n_buckets - 1 else max_length + 1
      self.buckets[i] = torch.tensor([idx for idx, length in enumerate(lengths) if lower_bound <= length < upper_bound], 
                                     dtype=torch.int, device='cpu')
    
    # Remove empty buckets
    self.buckets = {k: v for k, v in self.buckets.items() if len(v) > 0}
    
    if self.shuffle:
      for bucket_size in self.buckets.keys():
        self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]
    
    self.batches = []
    for bucket in self.buckets.values():
      curr_bucket = torch.split(bucket, self.batch_size)
      if len(curr_bucket) > 1 and self.drop_last:
        if len(curr_bucket[-1]) < len(curr_bucket[-2]):
          curr_bucket = curr_bucket[:-1]
      self.batches += curr_bucket
    
    self.length = len(self.batches)
    
    if self.shuffle:
      random.shuffle(self.batches)
  
  def __iter__(self):
    for i in range(self.length):
      yield self.batches[i].tolist()
  
  def __len__(self):
    return self.length

class DistributedOneDimKBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, lengths, n_buckets=10, shuffle=True, batch_size=32, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
        
        self.batch_size = batch_size
        self.lengths = lengths
        
        # Calculate bucket boundaries
        min_length = min(lengths)
        max_length = max(lengths)
        bucket_size = (max_length - min_length) // n_buckets
        
        # Create buckets
        self.buckets = {}
        for i in range(n_buckets):
            lower_bound = min_length + i * bucket_size
            upper_bound = lower_bound + bucket_size if i < n_buckets - 1 else max_length + 1
            self.buckets[i] = torch.tensor([idx for idx, length in enumerate(lengths) if lower_bound <= length < upper_bound], 
                                           dtype=torch.int, device='cpu')
        
        # Remove empty buckets
        self.buckets = {k: v for k, v in self.buckets.items() if len(v) > 0}
        
        self.shuffle_buckets()
        self.make_batches()
    
    def shuffle_buckets(self):
        if self.shuffle:
            for bucket in self.buckets.values():
                bucket[torch.randperm(len(bucket))]
    
    def make_batches(self):
        self.batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            self.batches += curr_bucket
        
        if self.shuffle:
            random.shuffle(self.batches)
    
    def __iter__(self):
        self.shuffle_buckets()
        self.make_batches()
        
        indices = torch.cat(self.batches)
        indices = indices[self.rank:len(indices):self.num_replicas]
        
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.dataset) // self.num_replicas


class MultimodalTokenDatasetMaker():
  def __init__(
    self,
    data_path: list[list[str]],
    data_dir: str,
    metadata_dir,
    n_codebook,
    codebook_size,
    max_seq_len,
    max_pt_x_len,
    num_special_tokens,
    image_height,
    image_compress_factor,
    midi_max_shift=1001,
    lmx_vocab_path='vocab/lmx_vocab_singletoken_asap.txt',
    in_modal_type=('lmx', 'midi', 'pt', 'dac'),
    out_modal_type=('lmx', 'midi', 'pt', 'dac'),
    debug=False,
    preload_data=False,
    modal_direction:str='bi',
    num_measure_to_slice: int=4,
    midi_slice_len: float=10.0,
    tps: int=100,
    out_pt_height_token: bool=False
  ):
    self.data_path = data_path
    self.data_dir = Path(data_dir)
    self.metadata_dir = metadata_dir
    self.n_codebook = n_codebook
    self.max_seq_len = max_seq_len
    self.max_pt_x_len = max_pt_x_len
    self.dataset_class = MultimodalTokenDataset
    self.in_modal_type = in_modal_type
    self.out_modal_type = out_modal_type
    self.total_modal_type = list(set(in_modal_type + out_modal_type))

    self.out_pt_height_token = out_pt_height_token

    self.preload_data = preload_data
    self.modal_direction = modal_direction
    self.num_measure_to_slice = num_measure_to_slice
    self.midi_slice_len = midi_slice_len

    self.in_idx_handler, self.out_idx_handler = self.get_vocab(num_special_tokens, n_codebook, codebook_size, max_seq_len, max_pt_x_len, image_height, image_compress_factor, midi_max_shift, tps, lmx_vocab_path)

    train_data_pairs = {}
    valid_data_pairs = {}
    test_data_pairs = {}
    self.sub_data_dirs = {}
    self.weight_by_dataset = {}
    for dataset_name, dataset_dir, modality_types, weight_of_dataset, (min_iter, max_iter) in data_path:
      json_fn = self.metadata_dir / f'{dataset_name}.json'
      if debug:
        debug_json_fn = self.metadata_dir / f'{dataset_name}_debug.json'
        if debug_json_fn.exists(): json_fn = debug_json_fn
      if not json_fn.exists() and json_fn.with_suffix('.json.gz').exists():
        json_fn = json_fn.with_suffix('.json.gz')
      opener = gzip.open if json_fn.suffix == '.gz' else open
      with opener(json_fn, 'rt') as f:
        pair_data = json.load(f)
        train_data_pairs[dataset_name] = self._filter_pair_by_length(pair_data['train'], modality_types)
        valid_data_pairs[dataset_name] = self._filter_pair_by_length(pair_data['valid'], modality_types)
        test_data_pairs[dataset_name] = self._filter_pair_by_length(pair_data['test'], modality_types)
        self.sub_data_dirs[dataset_name] = dataset_dir
        self.weight_by_dataset[dataset_name] = {'max_weight': weight_of_dataset, 'min_iter': min_iter, 'max_iter': max_iter}
    

    if debug:
      random.seed(0)
      for dataset_name in train_data_pairs.keys():
        random.shuffle(train_data_pairs[dataset_name])
        train_data_pairs[dataset_name] = train_data_pairs[dataset_name][:50]
        random.shuffle(valid_data_pairs[dataset_name])
        valid_data_pairs[dataset_name] = valid_data_pairs[dataset_name][:50]
        random.shuffle(test_data_pairs[dataset_name])
        test_data_pairs[dataset_name] = test_data_pairs[dataset_name][:50]
    self.train_data_pairs = train_data_pairs
    self.valid_data_pairs = valid_data_pairs
    self.test_data_pairs = test_data_pairs
    
    self.train_data_pairs = self._convert_to_absolute_path(self.train_data_pairs)
    self.valid_data_pairs = self._convert_to_absolute_path(self.valid_data_pairs)
    self.test_data_pairs = self._convert_to_absolute_path(self.test_data_pairs)

  def _filter_pair_by_length(self, list_of_pairs: list[dict[str, Path]], modality_types: list[str]):
    prototype_pair = list_of_pairs[0]
    keys = prototype_pair.keys()
    check_pt = 'pt_len' in keys
    check_dac = 'dac_len' in keys
    check_dac_exists = (sorted(modality_types) == ['dac', 'lmx']) and 'npy' in keys # ASAP but only using lmx and dac

    for pair in list_of_pairs: # change every 'npy' key to 'midi'
      if 'npy' in pair.keys():
        pair['midi'] = pair.pop('npy')
    
    if not check_pt and not check_dac and not check_dac_exists:
      return list_of_pairs
    
    filtered_pairs = []
    for pair in list_of_pairs:
      if check_pt and pair['pt_len'] + len(pair['pt'])-1 + 2 > self.max_seq_len['pt']:
        continue
      if check_dac and pair['dac_len'] + 2 > self.max_seq_len['dac']:
        continue
      included_keys = [key for key in pair.keys() if key in modality_types]
      if len(included_keys) < 2: # This happnes for ASAP sample without linked audio
        continue
      new_pair = {key: pair[key] for key in modality_types}
      if check_dac_exists:
        new_pair['midi_info'] = pair['midi']
      filtered_pairs.append(new_pair)
    return filtered_pairs
  
  def _convert_to_absolute_path(self, dict_of_pairs):
    for dataset_name in dict_of_pairs.keys():
      for pair in dict_of_pairs[dataset_name]:
        for k in pair.keys():
          if isinstance(pair[k], str) and '/' in pair[k]: # to exclude the yt_id
            pair[k] = self.data_dir / self.sub_data_dirs[dataset_name] / pair[k]
          elif isinstance(pair[k], list):
            pair[k] = [self.data_dir / self.sub_data_dirs[dataset_name] / p for p in pair[k]]
    return dict_of_pairs
  
  # @classmethod
  def get_vocab(self, num_special_tokens: int, 
                n_codebook: int, 
                codebook_size: int, 
                max_seq_len: dict, 
                max_pt_x_len: int,
                image_height: int, 
                image_compress_factor: int, 
                midi_max_shift: int,
                tps: int=100,
                lmx_vocab_path: str='vocab/lmx_vocab_singletoken_asap.txt'):
    if 'lmx' in self.total_modal_type:
      lmx_vocab = LMXVocab(
        vocab_txt_fn = lmx_vocab_path,
        num_special_tokens = num_special_tokens
      )

    if 'pt' in self.total_modal_type:
      img_vocab = RVQVocab(codebook_size, 
                        num_special_tokens=num_special_tokens + 1,
                        token_height=image_height//image_compress_factor, 
                        n_codebook=n_codebook)

    if 'dac' in self.total_modal_type: 
      dac_vocab = RVQVocab(codebook_size,
                        num_special_tokens=num_special_tokens, 
                        n_codebook=n_codebook)
    
    if 'midi' in self.total_modal_type:
      midi_vocab = NoteEventTokenizer(max_shift_steps=midi_max_shift, tps=tps,
                                    max_length=max_seq_len['midi']-2) # -2 for sos/eos tokens

    in_vocab_dict = {}
    for modal in self.in_modal_type:
      if modal == 'lmx':
        in_vocab_dict['lmx'] = lmx_vocab
      elif modal == 'midi':
        in_vocab_dict['midi'] = midi_vocab
      elif modal == 'pt':
        in_vocab_dict['pt'] = img_vocab
      elif modal == 'dac':
        in_vocab_dict['dac'] = dac_vocab
    
    in_idx_handler = TokenIdxHandler(in_vocab_dict, max_seq_len, max_pt_x_len)

    if self.in_modal_type != self.out_modal_type:
      out_vocab_dict = {}
      for modal in self.out_modal_type:
        if modal == 'lmx':
          out_vocab_dict['lmx'] = lmx_vocab
        elif modal == 'midi':
          out_vocab_dict['midi'] = midi_vocab
        elif modal == 'pt':
          out_vocab_dict['pt'] = img_vocab
        elif modal == 'dac':
          out_vocab_dict['dac'] = dac_vocab
        
      out_idx_handler = TokenIdxHandler(out_vocab_dict, max_seq_len, max_pt_x_len, out_pt_height_token=self.out_pt_height_token)
    else:
      out_idx_handler = in_idx_handler

    return in_idx_handler, out_idx_handler
  
  def get_datasets(self, test_set_only=False):
    if test_set_only:
      return self.dataset_class(self.test_data_pairs, self.max_seq_len, (self.in_idx_handler, self.out_idx_handler), self.in_modal_type, self.out_modal_type, is_valid=True, modal_direction=self.modal_direction, num_measure_to_slice=self.num_measure_to_slice, midi_slice_len=self.midi_slice_len, out_pt_height_token=self.out_pt_height_token)
    train_dataset = self.dataset_class(self.train_data_pairs, 
                                       self.max_seq_len, 
                                       (self.in_idx_handler, self.out_idx_handler), 
                                       self.in_modal_type, 
                                       self.out_modal_type, 
                                       is_valid=False, 
                                       weight_by_dataset=self.weight_by_dataset, 
                                       preload_data=self.preload_data, 
                                       modal_direction=self.modal_direction,
                                       num_measure_to_slice=self.num_measure_to_slice,
                                       midi_slice_len=self.midi_slice_len,
                                       )
    valid_dataset = self.dataset_class(self.valid_data_pairs, 
                                       self.max_seq_len, 
                                       (self.in_idx_handler, self.out_idx_handler), 
                                       self.in_modal_type, 
                                       self.out_modal_type, 
                                       is_valid=True, 
                                       preload_data=self.preload_data, 
                                       modal_direction=self.modal_direction,
                                       num_measure_to_slice=self.num_measure_to_slice,
                                       midi_slice_len=self.midi_slice_len,
                                       )
    test_dataset = self.dataset_class(self.test_data_pairs, 
                                       self.max_seq_len, 
                                       (self.in_idx_handler, self.out_idx_handler), 
                                       self.in_modal_type, 
                                       self.out_modal_type, 
                                       is_valid=True, 
                                       preload_data=self.preload_data, 
                                       modal_direction=self.modal_direction,
                                       num_measure_to_slice=self.num_measure_to_slice,
                                       midi_slice_len=self.midi_slice_len,
                                       )

    return train_dataset, valid_dataset, test_dataset
    
  def get_dacs_tok_pairs(self, data_dir:Path):
    dac_files = sorted(list(data_dir.rglob(f'*.dac')))
    pairs = []
    for dac_fn in dac_files:
      pairs.append({'dac': dac_fn, 'pt': dac_fn.with_suffix('.pt')})
    return pairs

  def get_specific_testset_loader(self, dataset_name: str, in_modal: str, out_modal: str, batch_size: int=16, use_valid: bool=False, use_train: bool=False, shuffle: bool=False):
    # test_dataset = self.get_datasets(test_set_only=True)
    if use_valid:
      path_pairs = {dataset_name: self.valid_data_pairs[dataset_name]}
    elif use_train:
      path_pairs = {dataset_name: self.train_data_pairs[dataset_name]}
    else:
      path_pairs = {dataset_name: self.test_data_pairs[dataset_name]}
    if dataset_name == 'asap' and 'dac' in [in_modal, out_modal]:
      path_pairs['asap'] = [pair for pair in path_pairs['asap'] if 'dac' in pair.keys()]
    test_dataset = TestDataset(path_pairs, self.max_seq_len, (self.in_idx_handler, self.out_idx_handler), self.in_modal_type, self.out_modal_type, is_valid=True, modal_direction=self.modal_direction, num_measure_to_slice=self.num_measure_to_slice, midi_slice_len=self.midi_slice_len)
    test_dataset.set_inout_modal(in_modal, out_modal)
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: multimodal_collate_fn(x, self.n_codebook))

class MultimodalTokenDataset():
  modal_order = {'pt': 0, 'lmx': 1, 'midi': 2, 'dac': 3}
  def __init__(self,
               data_path_pairs: list[dict[Path, Path]],
               max_length: dict,
               idx_handlers: tuple[TokenIdxHandler, TokenIdxHandler],
               in_modal_type: list[str],
               out_modal_type: list[str],
               dac_hop_sec: float=50,
               dac_sr: float=86.1328125,
               midi_slice_len: float=10,
               weight_by_dataset: dict[str, dict[str, float]]=None,
               is_valid: bool=False,
               preload_data: Optional[Union[bool, str]]=False,
               modal_direction:str='bi',
               num_measure_to_slice: int=4,
               ):
    super().__init__()

    self.max_length = max_length
    self.in_idx_handler, self.out_idx_handler = idx_handlers

    # Track starting indices for each dataset
    curr_idx = 0
    self.dataset_names = list(data_path_pairs.keys())
    self.dataset_start_indices = {}
    for dataset_name in data_path_pairs.keys():
        self.dataset_start_indices[dataset_name] = curr_idx
        curr_idx += len(data_path_pairs[dataset_name])

    # Create path_pairs list with dataset names
    self.path_pairs = [(dataset_name, pair) for dataset_name in data_path_pairs.keys() for pair in data_path_pairs[dataset_name]]
    self.num_sample_with_dataset_name = [ (dataset_name, len(data_path_pairs[dataset_name])) for dataset_name in data_path_pairs.keys()]
    self.in_modal_types = self.in_idx_handler.vocab_keys
    self.out_modal_types = self.out_idx_handler.vocab_keys
    if (type(preload_data) == bool and preload_data) or preload_data == 'all':
      self.data = self.load_all_data(self.path_pairs)
    elif preload_data == 'midi_only':
      self.data = {}
      self.load_midi_dataset(self.path_pairs)
    else:
      self.data = {}
    # self.check_data_exist()

    self.is_valid = is_valid
    self.in_modal_type = in_modal_type
    self.out_modal_type = out_modal_type
    self.dac_hop_sec = dac_hop_sec
    self.midi_slice_len = midi_slice_len
    self.dac_sr = dac_sr
    self.weight_by_dataset = weight_by_dataset
    if not is_valid:
      self._update_idx2weight(0)
    self.modal_direction = modal_direction
    self.num_measure_to_slice = num_measure_to_slice
    self.dac_len_sec = 60.0
      
    self.midi_max_shift = 0
    self.midi_tokenizer = None
    if 'midi' in self.in_modal_type:
      self.midi_max_shift = self.in_idx_handler.vocabs['midi'].max_shift_steps
      self.midi_tokenizer = self.in_idx_handler.vocabs['midi']
    elif 'midi' in self.out_modal_type:
      self.midi_max_shift = self.out_idx_handler.vocabs['midi'].max_shift_steps
      self.midi_tokenizer = self.out_idx_handler.vocabs['midi']
    if 'dac' in self.in_modal_type:
      self.max_dac_len = self.in_idx_handler.max_seq_len['dac']
    elif 'dac' in self.out_modal_type:
      self.max_dac_len = self.out_idx_handler.max_seq_len['dac']
    else:
      self.max_dac_len = 0
    

      
  def _update_idx2weight(self, iter:int=0):
    if self.weight_by_dataset is None:
      return None
    

    dataset_names = [x[0] for x in self.num_sample_with_dataset_name]
    cur_weight_by_dataset = {dataset_name: cal_weight_by_iter(float(self.weight_by_dataset[dataset_name]['max_weight']), 
                                                            iter, 
                                                            self.weight_by_dataset[dataset_name]['min_iter'], 
                                                            self.weight_by_dataset[dataset_name]['max_iter']) for dataset_name in dataset_names}
    idx_weights = []
    print(f"Weight by dataset: {cur_weight_by_dataset}")
    for dataset_name, num_sample in self.num_sample_with_dataset_name:
      idx_weights.extend([cur_weight_by_dataset[dataset_name]] * num_sample)
    self.idx2weight = torch.tensor(idx_weights)
    if torch.max(self.idx2weight) > 0:
      self.idx2weight /= min(1.0, torch.max(self.idx2weight))
    else:
      self.idx2weight = torch.ones_like(self.idx2weight)
    # print(f"Idx2weight: min:{torch.min(self.idx2weight)} max:{torch.max(self.idx2weight)} mean:{torch.mean(self.idx2weight)}")
    
  
  def check_data_exist(self):
    if torch.distributed.is_initialized():
      rank = torch.distributed.get_rank()
      world_size = torch.distributed.get_world_size()
    else:
      rank = 0
      world_size = 1
    return # Not implemented
    indices = range(len(self.path_pairs))[rank::world_size]
    for i in indices:
      dataset_name, pair = self.path_pairs[i]
      for k in pair.keys():
        if k not in self.modal_types:
          continue
        if isinstance(pair[k], list):
          pass
      
      
  def load_midi_dataset(self, path_pairs):
    if torch.distributed.is_initialized():
      rank = torch.distributed.get_rank()
      world_size = torch.distributed.get_world_size()
    else:
      rank = 0
      world_size = 1
    
    indices = range(len(path_pairs))[rank::world_size] # Assume this slice
    midi_indices = [i for i in indices if ('midi' in path_pairs[i][1].keys() or 'midi_info' in path_pairs[i][1].keys())]
    for i in tqdm(midi_indices, desc='Loading MIDI dataset'):
      dataset_name, pair = path_pairs[i]
      data = self.load_data(pair)
      self.data[i] = (dataset_name, data)
    print(f"Loaded {len(midi_indices)} samples including MIDI")
  
  def _load_lmx_tokens(self, lmx_path) -> tuple[torch.Tensor]:
    with open(lmx_path, 'r') as f:
      lmx_str = f.read()
    return lmx_str
    # # list of encoded tokens with sos and eos
    # if 'lmx' in self.in_modal_types:
    #   lmx_toks = self.in_idx_handler.vocabs['lmx'](lmx_str)
    # elif 'lmx' in self.out_modal_types:
    #   lmx_toks = self.out_idx_handler.vocabs['lmx'](lmx_str)
    # else:
    #   raise ValueError(f"Invalid modal types: 'lmx' not in in_modal:{self.in_modal_types} or out_modal:{self.out_modal_types}")
    
    # lmx_toks = torch.tensor(lmx_toks, dtype=torch.short)
    # return lmx_toks

  def _load_img_tokens(self, img_path, append_sos_eos=True) -> tuple[torch.Tensor]:
    if isinstance(img_path, list):
      return self._load_list_of_crop_img_tokens(img_path)
    img = torch.load(img_path, map_location='cpu', weights_only=True)
    img = img.to(torch.short)
    return img


  def _load_list_of_crop_img_tokens(self, img_path_list):
    list_of_img_tokens = [torch.load(p, map_location='cpu', weights_only=True).to(torch.short) for p in img_path_list]
    return list_of_img_tokens

  
  def _load_dac_tokens(self, dac_path, append_sos_eos=True) -> tuple[torch.Tensor]:
    if isinstance(dac_path, list):
      return self._load_list_of_crop_dac_tokens(dac_path)
    tokens = DACFile.load(dac_path).codes
    tokens = tokens.transpose(1,2).to(torch.short) # (n_tokens, n_codebook, n_timesteps)
    assert tokens.ndim == 3, f"Tokens shape: {tokens.shape}"
    return tokens
    # add sos and eos tokens
    tokens, token_height, dac_pos = self.idx_handler(tokens, 'dac', append_sos_eos=append_sos_eos)
    return tokens, token_height, dac_pos

  def _load_list_of_crop_dac_tokens(self, dac_path_list):
    return [self._load_dac_tokens(p, append_sos_eos=False) for p in dac_path_list], [float(p.stem.split('_')[-1]) for p in dac_path_list]
  
  def _load_midi_tokens(self, midi_path) -> tuple[torch.Tensor]:
    note_event_np = np.load(midi_path, allow_pickle=True).tolist()
    return note_event_np
  
  def load_all_data(self, path_pairs: list[tuple[str, dict[Path, Path]]]):
    data = {}
    if torch.distributed.is_initialized():
      rank = torch.distributed.get_rank()
      world_size = torch.distributed.get_world_size()
      
      indices = list(range(len(path_pairs)))[rank::world_size]
    else:
      indices = list(range(len(path_pairs)))
    
    for i in tqdm(indices, desc='Loading datasets'):
      dataset_name, pairs = path_pairs[i]
      data[i] = (dataset_name, self.load_data(pairs))
        
    # data = defaultdict(list)
    # for dataset_name in tqdm(data_path_pairs.keys(), desc='Loading Sub datasets'):
    #   pairs = data_path_pairs[dataset_name]
    #   num_valid_pairs = 0
    #   for pair in tqdm(pairs, desc='Loading pairs'):
    #     loaded_pair = self.load_data(pair)
    #     if loaded_pair is not None:
    #       data[dataset_name].append(loaded_pair)
    #       num_valid_pairs += 1
    #   print(f"Number of valid pairs for {dataset_name}: {num_valid_pairs}. Filtered {len(pairs) - num_valid_pairs} pairs")
    # data = dict(data)
    return data
  
  def load_data(self, pair: dict[Path, Path]):
    new_pair = {}
    for k in pair.keys():
      match k:
        case 'lmx':
          new_pair['lmx'] = self._load_lmx_tokens(pair[k])
        case 'dac':
          new_pair['dac'] = self._load_dac_tokens(pair[k])
        case 'pt':
          new_pair['pt'] = self._load_img_tokens(pair[k])
        case 'midi':
          new_pair['midi'] = self._load_midi_tokens(pair[k])
        case 'midi_info':
          new_pair['midi_info'] = self._load_midi_tokens(pair[k])
    # for k, v in new_pair.items():
    #   # if isinstance(v, tuple) and isinstance(v[0], torch.Tensor) and v[0].shape[-2] > self.max_length[k]:
    #   if isinstance(v, torch.Tensor):
    #     token_len = v.shape[-2] * v.shape[-3] if k == 'pt' else len(v)
    #     if token_len > self.max_length[k]-2:
    #       # this is a hack to ignore pairs that are too long
    #       # It will ignore long dac files with MIDI pairs, as they are list
    #       return None
    #   if k == 'pt' and isinstance(v, list) and sum([t.shape[-2] for t in v]) > self.max_length[k]-4: # considering the sep tokens
    #     return None
    return new_pair
  
  def _slice_dac_by_start_end_sec(self, dac_tok_list, dac_start_sec_list, start_sec, end_sec):
    dac_slice_idx = min(int(start_sec // self.dac_hop_sec), len(dac_tok_list)-1)
    offset_within_slice = start_sec - dac_start_sec_list[dac_slice_idx]
    
    corresp_dac_token = dac_tok_list[dac_slice_idx]
    corresp_dac_start_idx = int(offset_within_slice * self.dac_sr)
    corresp_dac_end_idx = corresp_dac_start_idx + int((end_sec - start_sec) * self.dac_sr)
    dac_slice = corresp_dac_token[:, corresp_dac_start_idx:corresp_dac_end_idx]
    return dac_slice
  
  
  def _slice_midi_pairs(self, data:dict, start_sec:Optional[float]=None, add_random_margin:bool=True):
    token_len = 1000000
    slice_len_sec = self.midi_slice_len - random.uniform(0, 1.0) if add_random_margin else self.midi_slice_len
    dac_tok_list, dac_start_sec_list = data['dac']
    npy_tokens = data['midi']
    duration = npy_tokens['duration_sec']
    # fix_start_sec = start_sec is not None
    if start_sec is None:
      start_sec = random.uniform(0, duration - slice_len_sec)
    while token_len + 2 > self.max_length['midi']: # +2 for sos/eos tokens
      # if not fix_start_sec:
      end_sec = start_sec + slice_len_sec
      dac_slice = self._slice_dac_by_start_end_sec(dac_tok_list, dac_start_sec_list, start_sec, end_sec)
      # dac_slice_idx = int(start_sec // self.dac_hop_sec)
      # offset_within_slice = start_sec - dac_start_sec_list[dac_slice_idx]
      # corresp_dac_token = dac_tok_list[dac_slice_idx]
      # corresp_dac_start_idx = int(offset_within_slice * self.dac_sr)
      # corresp_dac_end_idx = corresp_dac_start_idx + int(slice_len_sec * self.dac_sr)
      # dac_slice = corresp_dac_token[:, corresp_dac_start_idx:corresp_dac_end_idx]
      
      note_events = npy_tokens['note_events']
      note, tied, start_time = slice_note_events_and_ties(note_events, start_sec, end_sec)
      tokens = self.midi_tokenizer.encode(note, tie_note_events=tied, start_time=start_time, end_time=end_sec)
      
      token_len = len(tokens)
      if token_len + 2 > self.max_length['midi']: # +2 for sos/eos tokens
        slice_len_sec -= random.uniform(1.0, 2.0)
        slice_len_sec = max(slice_len_sec, 0.2)
      else:
        break
    assert dac_slice.ndim == 3, f"DAC slice ndim has to be 3, but got {dac_slice.shape}"
    return {'dac': dac_slice, 'midi': tokens}    
    # tokens, token_height, midi_pos = self.idx_handler(tokens, 'midi')
    # return {'dac': (dac_slice, dac_token_height, pos), 'midi': (tokens, token_height, midi_pos)}
  
  
  def _slice_score_midi_pairs(self, data:dict, start_measure:int=None, modal_types:list[str]=None):
    lmx_str = data['lmx']
    match_lmx_len_limit = False
    num_measure_to_slice = self.num_measure_to_slice
    measure_boundaries = get_measure_boundary_from_lmx(lmx_str)
    num_total_measures = len(measure_boundaries)
    fix_start_measure = start_measure is not None
    patience = 0 
    npy_tokens = data['midi'] if 'midi' in data.keys() else data['midi_info'] # 'midi_info' is a bypass to avoid being selected in modal selection
    if 'midi' in modal_types:
      match_midi_len_limit = False
      match_midi_max_shift = False
      match_dac_len_limit = True
    elif 'dac' in modal_types:
      match_dac_len_limit = False
      match_midi_len_limit = True
      match_midi_max_shift = True
    else:
      raise ValueError(f"Invalid modal types: {modal_types}")
      

    while not match_lmx_len_limit or not match_dac_len_limit or not match_midi_len_limit or not match_midi_max_shift:
      if patience > 5:
        fix_start_measure = False
      if not fix_start_measure:
        start_measure = random.randint(0, num_total_measures-num_measure_to_slice)
      start_sec, end_sec = slice_by_measure(npy_tokens['measure_map'], start_measure, num_measure_to_slice, margin=0.1)
      if start_sec is None:
        patience += 1
        continue
      
      if 'dac' in modal_types:
        if end_sec - start_sec + 0.05 > self.max_dac_len / self.dac_sr: # add 0.05 because of slice margin 
          num_measure_to_slice -= 1
          match_dac_len_limit = False
          patience += 1
          continue
        else:
          match_dac_len_limit = True
      elif 'midi' in modal_types:
        if end_sec - start_sec + 0.05 > (self.midi_max_shift-1) / 100: # add 0.05 because of slice margin 
          num_measure_to_slice -= 1
          match_midi_max_shift = False
          patience += 1
          continue
        else:
          match_midi_max_shift = True
      sliced_lmx = lmx_str.split(' ')[measure_boundaries[start_measure]:measure_boundaries[start_measure + num_measure_to_slice]]
      if len(sliced_lmx) + 2 > self.max_length['lmx']:
        match_lmx_len_limit = False
        num_measure_to_slice -= 1
        patience += 1
        continue
      else:
        match_lmx_len_limit = True
      if 'midi' in modal_types:
        note, tied, start_time = slice_note_events_and_ties(npy_tokens['note_events'], start_sec, end_sec, )
        tokens = self.midi_tokenizer.encode(note, tie_note_events=tied, start_time=start_time)
        if len(tokens) + 2 > self.max_length['midi']:
          match_midi_len_limit = False
          num_measure_to_slice -= 1
          patience += 1
          continue
        else:
          match_midi_len_limit = True
      num_measure_to_slice = max(1, num_measure_to_slice)
      patience += 1

    sliced_lmx = ' '.join(sliced_lmx)
    if 'dac' in modal_types:
      dac_tok_list, dac_start_sec_list = data['dac']
      offset = npy_tokens['audio_start'] if 'audio_start' in npy_tokens.keys() else 0
      dac_slice = self._slice_dac_by_start_end_sec(dac_tok_list, dac_start_sec_list, start_sec+offset, end_sec+offset)
      return {'lmx': sliced_lmx, 'dac': dac_slice}
    else:
      return {'lmx': sliced_lmx, 'midi': tokens}
  
  def make_piece_dac_slices(self, path_pair, slice_len:float=7.0, max_len_sec:float=None):
    data = self.load_data(path_pair)
    entire_duration = data['dac'][1][-1] + self.dac_len_sec
    if max_len_sec is not None:
      entire_duration = max_len_sec
    n_slices = math.ceil(entire_duration / slice_len)
    
    dac_slices = []
    for i in range(n_slices):
      start_sec = i * slice_len
      end_sec = start_sec + slice_len
      dac_slice = self._slice_dac_by_start_end_sec(data['dac'][0], data['dac'][1], start_sec, end_sec)
      dac_slice, _, in_pos = self.in_idx_handler(dac_slice, 'dac')
      if dac_slice.shape[0] > 1:
        dac_slice = dac_slice[dac_slice.shape[0]//2]
      dac_slices.append((dac_slice, in_pos))
    return dac_slices

  def select_in_out_modal(self, data, idx):
    modal_types = sorted(list(data.keys())) # ['dac', 'pt', 'midi', 'lmx'] # currently only two modalities are included in a single sample
    if 'midi_info' in data.keys():
      modal_types.remove('midi_info')
    if self.modal_direction == 'bi':
      if self.is_valid:
        in_modal, out_modal = modal_types[idx % 2], modal_types[(idx+1) % 2]
      else:
        in_modal, out_modal = random.sample(modal_types, 2)
    elif self.modal_direction == 'omr':
      # only go to pt -> lmx -> midi -> dac
      # always select leftmost modal as in_modal
      match modal_types:
        case ['lmx', 'pt']:
          in_modal = 'pt'
          out_modal = 'lmx'
        case ['dac', 'pt']:
          in_modal = 'pt'
          out_modal = 'dac'
        case ['dac', 'midi']:
          in_modal = 'midi'
          out_modal = 'dac'
        case ['lmx', 'midi']:
          in_modal = 'lmx'
          out_modal = 'midi'
        case ['dac', 'lmx', 'midi']:
          in_modal = 'lmx'
          out_modal = random.choice(['midi', 'dac'])
        case ['dac', 'lmx']:
          in_modal = 'lmx'
          out_modal = 'dac'
        case _:
          raise ValueError(f"Invalid modal types: {modal_types}") 

      # sorted_modal_types = sorted(modal_types, key=lambda x: self.modal_order[x])
      # in_modal_idx = random.randint(0, len(sorted_modal_types)-2)
      # out_modal_idx = random.randint(in_modal_idx+1, len(sorted_modal_types)-1)
      # in_modal = sorted_modal_types[in_modal_idx]
      # out_modal = sorted_modal_types[out_modal_idx]
    elif self.modal_direction == 'amt':
      # only go to dac -> midi -> lmx -> pt
      # always select rightmost modal as in_modal
      match modal_types:
        case ['lmx', 'pt']:
          in_modal = 'lmx'
          out_modal = 'pt'
        case ['dac', 'pt']:
          in_modal = 'dac'
          out_modal = 'pt'
        case ['dac', 'midi']:
          in_modal = 'dac'
          out_modal = 'midi'
        case ['lmx', 'midi']:
          in_modal = 'midi'
          out_modal = 'lmx'
        case ['dac', 'lmx', 'midi']:
          in_modal = random.choice(['midi', 'dac'])
          out_modal = 'lmx'
        case ['dac', 'lmx']:
          in_modal = 'dac'
          out_modal = 'lmx'
        case _:
          raise ValueError(f"Invalid modal types: {modal_types}")      
      # sorted_modal_types = sorted(modal_types, key=lambda x: self.modal_order[x])
      # out_modal_idx = random.randint(0, len(sorted_modal_types)-2)
      # in_modal_idx = random.randint(out_modal_idx+1, len(sorted_modal_types)-1)
      # in_modal = sorted_modal_types[in_modal_idx]
      # out_modal = sorted_modal_types[out_modal_idx]
      
    # Fix for single modal type
    if len(self.in_modal_type) == 1:
      in_modal = self.in_modal_type[0]
    if len(self.out_modal_type) == 1:
      out_modal = self.out_modal_type[0]
      if len(modal_types) == 2:
        in_modal = modal_types[modal_types.index(out_modal)-1]
    return in_modal, out_modal
  
  def __len__(self):
    # return len(self.data)
    # return sum([len(v) for v in self.data.values()])
    return len(self.path_pairs)
  
  def __getitem__(self, idx):
    # randomly select a dataset
    # dataset_name = random.choice(list(self.data.keys()))
    # idx = random.randint(0, len(self.data[dataset_name])-1)
    # data = self.data[dataset_name][idx]
    if idx in self.data:
      dataset_name, data = self.data[idx]
    else:
      dataset_name, data_pair = self.path_pairs[idx]
      data = self.load_data(data_pair)
      self.data[idx] = (dataset_name, data)
    in_modal, out_modal = self.select_in_out_modal(data, idx)
    if dataset_name in ['musicnet', 'maestro', 'slakh', 'musicnet_ogdac', 'maestro_ogdac', 'slakh_ogdac']:
      if self.is_valid:
        data = self._slice_midi_pairs(data, start_sec=10, add_random_margin=False)
      else:
        data = self._slice_midi_pairs(data, add_random_margin=in_modal == 'midi')
    elif dataset_name == 'asap':
      if self.is_valid:
        data = self._slice_score_midi_pairs(data, start_measure=0, modal_types=[in_modal, out_modal])
      else:
        data = self._slice_score_midi_pairs(data, modal_types=[in_modal, out_modal])
      
    in_data, in_token_height, in_pos = self.in_idx_handler(data[in_modal], in_modal)
    out_data, out_token_height, out_pos = self.out_idx_handler(data[out_modal], out_modal)
    
    # TODO: generalize this
    if in_modal == 'pt':
      if self.is_valid:
        x_shift_idx = in_data.shape[-4]//2
        y_shift_idx = in_data.shape[-3]//2
        if in_data.ndim == 5: # olimpic augmented synthetic data
          aug_idx = 0
      else:
        x_shift_idx = random.randint(0, in_data.shape[-4]-1)
        y_shift_idx = random.randint(0, in_data.shape[-3]-1)
        if in_data.ndim == 5: # olimpic augmented synthetic data
          aug_idx = random.randint(0, in_data.shape[-5]-1)
      if in_data.ndim == 5: # olimpic augmented synthetic data
        in_data = in_data[aug_idx, x_shift_idx, y_shift_idx] # 0 is non-augmented
      else:
        in_data = in_data[x_shift_idx, y_shift_idx]
    if out_modal == 'pt':
      if self.is_valid:
        x_shift_idx = out_data.shape[-4]//2
        y_shift_idx = out_data.shape[-3]//2
        if out_data.ndim == 5: # olimpic augmented synthetic data
          aug_idx = 0
      else:
        x_shift_idx = random.randint(0, out_data.shape[-4]-1)
        y_shift_idx = random.randint(0, out_data.shape[-3]-1)
        if out_data.ndim == 5: # olimpic augmented synthetic data
          aug_idx = 0
      if out_data.ndim == 5: # olimpic augmented synthetic data
        out_data = out_data[aug_idx, x_shift_idx, y_shift_idx] # 0 is non-augmented
      else:
        out_data = out_data[x_shift_idx, y_shift_idx]
      
    if in_modal == 'dac':
      if self.is_valid:
        aug_idx = min(in_data.shape[0]-1, in_data.shape[0]//2)
      else:
        aug_idx = random.randint(0, in_data.shape[0]-1)
      in_data = in_data[aug_idx]
    if out_modal == 'dac':
      if self.is_valid:
        aug_idx = min(out_data.shape[0]-1, out_data.shape[0]//2)
      else:
        aug_idx = random.randint(0, out_data.shape[0]-1)
      out_data = out_data[aug_idx]
    
    target_in = out_data[:-1].long()
    target_out = out_data[1:].long()
    
    target_in_pos = out_pos[:-1]
    # assert len(in_data) <= self.max_length[in_modal], f"In data length {len(in_data)} is larger than max length {self.max_length[in_modal]} for {in_modal}, idx {idx}"
    # assert len(target_in) <= self.max_length[out_modal], f"Target in data length {len(target_in)} is larger than max length {self.max_length[out_modal]} for {out_modal}, idx {idx}"

    return in_data.long(), target_in, target_out, torch.tensor([self.in_modal_types.index(in_modal), self.out_modal_types.index(out_modal)]), torch.tensor([in_token_height, out_token_height]), in_pos, target_in_pos


class TestDataset(MultimodalTokenDataset):
  def __init__(self, data_path_pairs: list[dict[Path, Path]], max_length: dict, idx_handlers: tuple[TokenIdxHandler, TokenIdxHandler], in_modal_type: list[str], out_modal_type: list[str], dac_hop_sec: float=50, dac_sr: float=86.1328125, midi_slice_len: float=10, weight_by_dataset: dict[str, dict[str, float]]=None, is_valid: bool=True, preload_data: Optional[Union[bool, str]]=False, modal_direction:str='bi', num_measure_to_slice: int=4):
    super().__init__(data_path_pairs, max_length, idx_handlers, in_modal_type, out_modal_type, dac_hop_sec, dac_sr, midi_slice_len, weight_by_dataset, is_valid, preload_data, modal_direction, num_measure_to_slice)
    self.in_modal = None
    self.out_modal = None
    assert self.is_valid, "TestDataset is only supported for validation"
    self.default_midi_start_sec = 10
    self.default_score_start_measure = 0
  
  def set_inout_modal(self, in_modal, out_modal):
    self.in_modal = in_modal
    self.out_modal = out_modal
    
  def __getitem__(self, idx):
    assert self.in_modal is not None and self.out_modal is not None, "In and out modal types must be set before calling __getitem__"
    in_modal, out_modal = self.in_modal, self.out_modal
    dataset_name, data_pair = self.path_pairs[idx]
    data = self.load_data(data_pair)
    if dataset_name in ['musicnet', 'maestro', 'slakh', 'musicnet_ogdac', 'maestro_ogdac', 'slakh_ogdac']:
      data = self._slice_midi_pairs(data, start_sec=self.default_midi_start_sec)
    elif dataset_name in ['asap', 'bpsd']:
      data = self._slice_score_midi_pairs(data, start_measure=self.default_score_start_measure, modal_types=[in_modal, out_modal])
      
    in_data, in_token_height, in_pos = self.in_idx_handler(data[in_modal], in_modal)
    out_data, out_token_height, out_pos = self.out_idx_handler(data[out_modal], out_modal)
    
    # TODO: generalize this
    if in_modal == 'pt':
      x_shift_idx = in_data.shape[-4]//2
      y_shift_idx = in_data.shape[-3]//2
      if in_data.ndim == 5: # olimpic augmented synthetic data
        aug_idx = 0
      if in_data.ndim == 5: # olimpic augmented synthetic data
        in_data = in_data[aug_idx, x_shift_idx, y_shift_idx] # 0 is non-augmented
      else:
        in_data = in_data[x_shift_idx, y_shift_idx]
    if out_modal == 'pt':
      x_shift_idx = out_data.shape[-4]//2
      y_shift_idx = out_data.shape[-3]//2
      if out_data.ndim == 5: # olimpic augmented synthetic data
        aug_idx = 0
      if out_data.ndim == 5: # olimpic augmented synthetic data
        out_data = out_data[aug_idx, x_shift_idx, y_shift_idx] # 0 is non-augmented
      else:
        out_data = out_data[x_shift_idx, y_shift_idx]
      
    if in_modal == 'dac':
      aug_idx = min(in_data.shape[0]-1, in_data.shape[0]//2)
      in_data = in_data[aug_idx]
    if out_modal == 'dac':
      aug_idx = min(out_data.shape[0]-1, out_data.shape[0]//2)
      out_data = out_data[aug_idx]
    
    target_in = out_data[:-1].long()
    target_out = out_data[1:].long()
    
    target_in_pos = out_pos[:-1]
    # assert len(in_data) <= self.max_length[in_modal], f"In data length {len(in_data)} is larger than max length {self.max_length[in_modal]} for {in_modal}, idx {idx}"
    # assert len(target_in) <= self.max_length[out_modal], f"Target in data length {len(target_in)} is larger than max length {self.max_length[out_modal]} for {out_modal}, idx {idx}"

    return in_data.long(), target_in, target_out, torch.tensor([self.in_modal_types.index(in_modal), self.out_modal_types.index(out_modal)]), torch.tensor([in_token_height, out_token_height]), in_pos, target_in_pos


class CustomDistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, idx2weight: Optional[torch.Tensor] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.idx2weight = idx2weight
        if self.idx2weight is not None:
          assert len(self.idx2weight) == len(self.dataset)
          self.expected_total_size = math.ceil(sum(self.idx2weight) / self.num_replicas)


    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if self.idx2weight is None:
          pass
        elif not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            assert len(indices) == self.total_size
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.shuffle:
          g = torch.Generator()
          g.manual_seed(self.seed + self.epoch)
          indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)]
        assert len(indices) == self.num_samples
        
        if self.idx2weight is not None:
          if isinstance(indices, list): indices = torch.tensor(indices)
          rand_u = torch.rand(len(indices))
          weight_of_indices = self.idx2weight[indices]
          filtered_indices = indices[rand_u < weight_of_indices]
          
          marginal_weight = 0.05
          oversample_th = 1.0
          oversample_added = False
          do_oversample = torch.max(weight_of_indices) > 1.0
          rand_u[weight_of_indices==0] = 1
          while do_oversample and len(filtered_indices) < self.expected_total_size and oversample_th < 10.0:
            new_indices = indices[(rand_u + oversample_th) < weight_of_indices]
            if len(new_indices) == 0:
              break
            filtered_indices = torch.cat([filtered_indices, new_indices])
            oversample_added = True
            oversample_th += 1.0
          if oversample_added and self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            filtered_indices = filtered_indices[torch.randperm(len(filtered_indices), generator=g)]
          while len(filtered_indices) < self.expected_total_size:
            rand_u[weight_of_indices==0] = 1
            new_indices = indices[(rand_u >= weight_of_indices) * (rand_u < weight_of_indices+marginal_weight)]
            if len(new_indices) == 0:
              # Randomly oversample remaining samples from filtered indices
              num_remaining = self.expected_total_size - len(filtered_indices)
              g = torch.Generator()
              g.manual_seed(self.seed + self.epoch)
              oversampled_indices = filtered_indices[torch.randint(len(filtered_indices), (num_remaining,), generator=g)]
              filtered_indices = torch.cat([filtered_indices, oversampled_indices])
              break
            filtered_indices = torch.cat([filtered_indices, new_indices[:self.expected_total_size-len(filtered_indices)]])
            marginal_weight += 0.05
          if len(filtered_indices) > self.expected_total_size:
            filtered_indices = filtered_indices[:self.expected_total_size]
          indices = filtered_indices.tolist()
          assert len(indices) == self.expected_total_size
        if isinstance(indices, torch.Tensor):
          indices = indices.tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch
