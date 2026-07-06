import time
import os
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict
import random
import math

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import torchaudio
import torchaudio.transforms as audioT

from pydub import AudioSegment
from midi2audio import FluidSynth

import cv2

from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import numpy as np
from einops import rearrange

import wandb

from .model_zoo import LatentScoreAMT, PianoRollAMT, LMX2VQAMT
from . import data_utils
from .data_utils import audio_token_collate_fn, pianoroll_collate_fn, lmx_vq_collate_fn, multimodal_collate_fn, CustomDistributedSampler, DistributedEvalSampler, MultimodalTokenDataset
from .vocab_utils import VQVocab, RVQVocab
from .lmx_utils import delinearize_lmx, render_xml_with_lilypond, render_xml_with_musescore
from .evaluation_utils import LayerPeeper, use_attn_weights, draw_attention_map, Evaluator
from .constants import *
from .yourmt3plus.model.spectrogram import get_spectrogram_layer_from_audio_cfg
from .yourmt3plus.model.ops import minmax_normalize
from .midi_utils.midi import note_event2midi
from .utils import set_seed
from .data_decode_utils import TensorDecoder
class BaseTrainer:
  def __init__(
    self,
    model,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    loss_fn, 
    train_set, 
    valid_set, 
    save_dir: str, 
    use_ddp: bool,
    bucket_config,
    use_fp16: bool,
    world_size: int,
    batch_size: int,
    gpu_id: int,
    config,
    wandb_run = None,
    num_workers: int = 4,
    start_iter: int = 0,
  ):
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    self.train_set = train_set
    self.valid_set = valid_set
    self.use_ddp = use_ddp
    self.world_size = world_size
    self.batch_size = batch_size
    self.gpu_id = gpu_id
    self.config = config
    self.start_iter = start_iter
    self.num_workers = num_workers
    # bucket params
    self.bucket_config = bucket_config

    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(exist_ok=True, parents=True)
    
    self.train_loader = self.generate_data_loader(train_set, shuffle=True, drop_last=False, collate_fn=self.collate_fn, iter=self.start_iter)
    self.valid_loader = self.generate_data_loader(valid_set, shuffle=False, drop_last=False, collate_fn=self.collate_fn, is_valid=True)

    if use_ddp:
      self.device = torch.device(f'cuda:{self.gpu_id}')
      self.model.to(self.device) 
      self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=False)
    else:
      self.device = config.train_params.device
      self.model.to(self.device)
    
    if use_fp16:
      self.use_fp16 = True
      self.scaler = torch.cuda.amp.GradScaler()
    else:
      self.use_fp16 = False
    
    if isinstance(self.loss_fn, torch.nn.Module):
      self.loss_fn = self.loss_fn.to(self.device)
    self.grad_clip = config.train_params.grad_clip
    
    self.num_cycles_for_inference = config.train_params.num_cycles_for_inference
    self.num_cycles_for_model_checkpoint = config.train_params.num_cycles_for_model_checkpoint
    self.iterations_per_training_cycle = config.train_params.iterations_per_training_cycle
    self.iterations_per_validation_cycle = config.train_params.iterations_per_validation_cycle
    
    self.make_log = config.general.make_log
    self.infer_and_log = config.general.infer_and_log
    self.num_inference = config.inference_params.n_inference
    self.log_train_metric = config.general.log_train_metric
    
    self.wandb_run = wandb_run
    
    self.best_valid_accuracy = 0
    self.best_valid_loss = 100

    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []

    self.audio_preprocess_fn = None
    
    self.set_save_out()
    # self.check_batch_size_fit_gpu()
    
  def set_save_out(self):
    if self.infer_and_log:
      self.valid_out_dir = self.save_dir.parent / 'valid_out'
      os.makedirs(self.valid_out_dir, exist_ok=True)
      
  def check_batch_size_fit_gpu(self):
    raise NotImplementedError("Subclasses must implement this method")

  def generate_data_loader(self, dataset, shuffle=False, drop_last=False, collate_fn=None, epoch=0, is_valid=False, iter:int=0) -> DataLoader:
    assert collate_fn is not None
    
    if self.use_ddp:
      if self.bucket_config.use_bucket:
        batch_sampler = getattr(data_utils, self.bucket_config.class_name)(
          num_replicas=self.world_size, 
          rank=self.gpu_id,
          lengths=dataset.get_data_lengths(),
          n_buckets=self.bucket_config.n_buckets,
          shuffle=shuffle,
          batch_size=self.batch_size,
          drop_last=drop_last,
        )
        return DataLoader(
          dataset,
          batch_size=1, 
          shuffle=False, 
          drop_last=drop_last,
          collate_fn=collate_fn,
          batch_sampler=batch_sampler,
          num_workers=self.num_workers,
          pin_memory=True
        )
      else:
        if is_valid:
          sampler = DistributedEvalSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.gpu_id,
            shuffle=False,
            seed=self.config.general.seed,
          )
        else:
          dataset._update_idx2weight(iter)
          sampler = CustomDistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.gpu_id,
            shuffle=shuffle,
            seed=self.config.general.seed,
          idx2weight=dataset.idx2weight,
        )
        sampler.set_epoch(epoch)
        
        prefetch_factor = 4 if self.num_workers > 0 else None
        multiprocessing_context = torch.multiprocessing.get_context('fork') if self.num_workers > 0 else None

        return DataLoader(
          dataset, 
          batch_size=self.batch_size, 
          shuffle=False, 
          drop_last=drop_last,
          collate_fn=collate_fn,
          sampler=sampler,
          num_workers=self.num_workers,
          prefetch_factor=prefetch_factor,
          multiprocessing_context=multiprocessing_context,
          pin_memory=True,
        )
    
    if self.bucket_config.use_bucket:
      batch_sampler = getattr(data_utils, self.bucket_config.class_name)(
        lengths=dataset.get_data_lengths(),
        n_buckets=self.bucket_config.n_buckets,
        shuffle=shuffle,
        batch_size=self.batch_size,
        drop_last=drop_last,
      )
      
      return DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False, 
        drop_last=drop_last,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
        num_workers=self.num_workers,
        pin_memory=True
      )
    else:
      return DataLoader(
        dataset,
        batch_size=self.batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=self.num_workers,
        pin_memory=True
      )

  def save_checkpoint(self, epoch, loss):
    scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else None
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scheduler_state_dict': scheduler_dict,
      'loss': loss,
    }
    torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')

  def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.gpu_id}')
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

  def train_by_num_iter(self, num_iters):
    if self.config.nn_params.compile:
      self.compile_model()
    generator = iter(self.train_loader)
    epoch = 0
    # if self.use_ddp:
    #   vocab = self.model.module.decoder.net.vocab
    # else:
    #   vocab = self.model.decoder.net.vocab
    validation_loss = math.inf
    for i in tqdm(range(self.start_iter, num_iters)):
      try:
        batch = next(generator)
      except StopIteration:
        epoch += 1
        self.train_loader = self.generate_data_loader(self.train_set, shuffle=True, drop_last=False, collate_fn=self.collate_fn, epoch=epoch, iter=i)
        generator = iter(self.train_loader)
        batch = next(generator)

      self.model.train()

      make_log = (i+1) % self.iterations_per_training_cycle == 0 and self.make_log  
      _, loss_dict = self._train_by_single_batch(batch, get_loss_dict=make_log)
      
      if make_log:
        loss_dict = self._rename_dict(loss_dict, 'train')
        self.wandb_run.log(loss_dict, step=i)
      
      if (i+1) % self.iterations_per_validation_cycle == 0:
        validation_loss = self.run_validate(i)
        # self.model.eval()
        # if isinstance(vocab, RVQVocab):
        #   validation_loss, num_correct_guess, num_correct_guess_dict, validation_metrics = self.validate()
        #   validation_metrics['accuracy'] = num_correct_guess
        #   for key in num_correct_guess_dict.keys():
        #     validation_metrics[f'accuracy_k{key}'] = num_correct_guess_dict[key]
        # else:
        #   validation_loss, validation_metrics = self.validate()
        # validation_metrics['loss'] = validation_loss
        # validation_metrics = self._rename_dict(validation_metrics, 'valid')
        # if self.make_log:
        #   self.wandb_run.log(validation_metrics, step=i)
      if (i+1) % (self.iterations_per_validation_cycle // 2) == 0:
        self.save_model(self.save_dir / 'last_checkpoint.pt', i, validation_loss)

      if (i+1) % (self.iterations_per_validation_cycle * self.num_cycles_for_inference) == 0 and self.infer_and_log and self.make_log:
        if self.config.general.seed is not None:
          set_seed(self.config.general.seed)
        self.inference_and_log(i, self.num_inference)
        if self.config.general.seed is not None:
          set_seed(self.config.general.seed)
      
      if torch.distributed.is_initialized():
        torch.distributed.barrier()
      if (i+1) % (self.iterations_per_validation_cycle * self.num_cycles_for_model_checkpoint) == 0:
        self.save_model(self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt', i, validation_loss)
        print(f"Checkpoint : {i}th iter : valid loss {validation_loss:.4f}", " : ", str(self.save_dir), " / ", f"iter{i}_loss{validation_loss:.4f}.pt")
    
    # save last checkpoint
    self.save_model(self.save_dir / f'iter{num_iters}_loss{validation_loss:.4f}.pt', num_iters, validation_loss)


  def _train_by_single_batch(self, batch, get_loss_dict=True):
    start_time = time.time()
    
    if self.audio_preprocess_fn is not None:
      batch = self.audio_preprocess_fn(batch)

    loss, _, loss_dict = self._get_loss_pred_from_single_batch(batch, get_loss_dict=get_loss_dict)
    
    if self.use_fp16:
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()
    
    self.optimizer.zero_grad()
    
    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.scheduler is not None:
      self.scheduler.step()
    
    loss_dict['time'] = time.time() - start_time
    loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
    
    return loss.item(), loss_dict

  def run_validate(self, i):
    self.model.eval()
    validation_loss, validation_metrics = self.validate()
    validation_metrics['loss'] = validation_loss
    if self.make_log:
      validation_metrics = self._rename_dict(validation_metrics, 'valid')
      self.wandb_run.log(validation_metrics, step=i)
    return validation_loss
    if torch.distributed.is_initialized():
      counter = torch.zeros((3,), device=self.device)
      counter[0] += validation_metrics['total_loss']
      counter[1] += validation_metrics['total_num_correct_guess']
      counter[2] += validation_metrics['total_num_tokens']  
      torch.distributed.reduce(counter, 0)

    if self.make_log:
      validation_metrics['total_loss'] = counter[0]
      validation_metrics['total_num_correct_guess'] = counter[1]
      validation_metrics['total_num_tokens'] = counter[2]
      validation_metrics['loss'] = (counter[0] / counter[2]).item()
      validation_metrics['accuracy'] = (counter[1] / counter[2]).item()
      validation_metrics.pop('total_loss')
      validation_metrics.pop('total_num_correct_guess')
      validation_metrics.pop('total_num_tokens')
      validation_metrics = self._rename_dict(validation_metrics, 'valid')    
      self.wandb_run.log(validation_metrics, step=i)
    return validation_loss

  def _get_loss_pred_from_single_batch(self, batch, get_loss_dict=True):
    raise NotImplementedError("Subclasses must implement this method")

  def validate(self):
    raise NotImplementedError("Subclasses must implement this method")

  def inference(self, audio_path):
    raise NotImplementedError("Subclasses must implement this method")

  def inference_and_log(self, n_iter, num_inference):
    raise NotImplementedError("Subclasses must implement this method")

  def _rename_dict(self, adict, prefix):
    return {f'{prefix}.{k}': v for k, v in adict.items()}

  def save_model(self, path, iter, loss):
    if self.use_ddp:
      if torch.distributed.get_rank() == 0:
        state_dict = self.model.module.state_dict()
      else:
        return # Do not save model if not the master process
    else:
      state_dict = self.model.state_dict()
    scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else None
    checkpoint = {
      'iter': iter,
      'model_state_dict': state_dict,
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scheduler_state_dict': scheduler_dict,
      'loss': loss,
    }
    if self.use_fp16:
      checkpoint['scaler_state_dict'] = self.scaler.state_dict()
    torch.save(checkpoint, path)
  
  def compile_model(self):
    if self.use_ddp:
      self.model.module.compile_model()
    else:
      self.model.compile_model()
  
  def uncompile_model(self):
    if self.use_ddp:
      self.model.module.uncompile_model()
    else:
      self.model.uncompile_model()

  
class MultimodalTrainer(BaseTrainer):
  def __init__(
    self,
    model,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    loss_fn, 
    train_set, 
    valid_set, 
    save_dir: str, 
    use_ddp: bool,
    bucket_config,
    use_fp16: bool,
    world_size: int,
    batch_size: int,
    gpu_id: int,
    config,
    rq_model,
    dac_model,
    wandb_run = None,
    num_workers: int = 4,
    start_iter: int = 0,
  ):
    self.rq_model = rq_model
    self.dac_model = dac_model
    self.fs = FluidSynth()
    self.in_vocab = model.in_vocab
    self.out_vocab = model.out_vocab

    self.collate_fn = self.collate_wrapper
    super().__init__(model, optimizer, scheduler, loss_fn, train_set, valid_set, save_dir, use_ddp, bucket_config, use_fp16, world_size, batch_size, gpu_id, config, wandb_run, num_workers, start_iter)
    self.modal_pairs_code, self.modal_pairs_map = self._prepare_all_possible_modal_pairs()
    self.decoder = TensorDecoder(config, self.in_vocab, self.out_vocab, Path(save_dir) / 'valid_out', device=f'cpu', vq_model=self.rq_model, dac_model=self.dac_model)
  def _prepare_all_possible_modal_pairs(self):
    # this is to gather validation metrics for each modal pair in DDP
    # modal_keys = list(set(self.in_vocab.vocab_keys + self.out_vocab.vocab_keys))
    in_modal_keys = self.in_vocab.vocab_keys
    out_modal_keys = self.out_vocab.vocab_keys
    modal_pairs = []
    for i in range(len(in_modal_keys)):
      for j in range(len(out_modal_keys)):
        if in_modal_keys[i] == out_modal_keys[j]:
          continue
        modal_pairs.append(f"{in_modal_keys[i]}-to-{out_modal_keys[j]}")
    keys2idx = {key: i for i, key in enumerate(modal_pairs)}
    return modal_pairs, keys2idx
  
  def collate_wrapper(self, batch):
    if hasattr(self.config.nn_params, 'compile') and self.config.nn_params.compile:
      return multimodal_collate_fn(batch, self.loss_fn.n_codebook, self.in_vocab.max_tok_len, self.out_vocab.max_tok_len)
    else:
      return multimodal_collate_fn(batch, self.loss_fn.n_codebook)

        
  # def check_batch_size_fit_gpu(self):
  #   vocab_size = self.train_set.idx_handler.vocab_size
  #   max_len = self.train_set.idx_handler.max_tok_len
  #   in_modal = torch.randint(0, vocab_size, (self.batch_size, max_len, self.train_set.n_codebook), device=self.device)
  #   target_in = torch.randint(0, vocab_size, (self.batch_size, max_len, self.train_set.n_codebook), device=self.device)
  #   target_out = torch.randint(0, vocab_size, (self.batch_size, max_len, self.train_set.n_codebook), device=self.device)
  #   in_mask = torch.ones((self.batch_size, max_len), device=self.device)
  #   modal_idx = torch.randint(0, len(self.train_set.in_modal_type), (self.batch_size, 2), device=self.device)
    
    

  def _get_loss_pred_from_single_batch(self, batch, get_acc=False, get_loss_dict=True):
    in_modal, in_mask, target_in, target_out, modal_idx, token_heights, in_pos, target_in_pos = batch['in_modal'], batch['in_mask'], batch['target_in'], batch['target_out'], batch['modal_idx'], batch['token_height'], batch['in_pos'], batch['target_in_pos']
    
    in_modal = in_modal.to(self.device)
    in_mask = in_mask.to(self.device)
    target_in = target_in.to(self.device)
    target_out = target_out.to(self.device)
    modal_idx = modal_idx.to(self.device)
    token_heights = token_heights.to(self.device)
    in_pos = in_pos.to(self.device)
    target_in_pos = target_in_pos.to(self.device)

    if self.use_fp16:
      with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = self.model(in_modal, in_mask, target_in, target_out, modal_idx, in_pos, target_in_pos)
        loss, loss_dict = self.loss_fn(logits, target_out, modal_idx, get_acc=get_acc, get_loss_dict=get_loss_dict)
    else:
      logits = self.model(in_modal, in_mask, target_in, target_out, modal_idx, in_pos, target_in_pos)
      loss, loss_dict = self.loss_fn(logits, target_out, modal_idx, get_acc=get_acc, get_loss_dict=get_loss_dict)
    
    return loss, logits, loss_dict
  
  @torch.inference_mode()
  def validate(self):
    total_loss = 0
    total_num_correct_guess = 0
    total_num_tokens = 0
    all_metrics = defaultdict(list)
    
    for batch in tqdm(self.valid_loader, leave=False):
      loss, logits, loss_dict = self._get_loss_pred_from_single_batch(batch, get_acc=True)

      total_loss += loss.item() * loss_dict['total_num_tokens']
      total_num_tokens += loss_dict['total_num_tokens']
      total_num_correct_guess += loss_dict['total_num_correct']

      for key, value in loss_dict.items():
        value = value.item() if isinstance(value, torch.Tensor) else value
        all_metrics[key].append(value)
  
    avg_loss = total_loss / total_num_tokens
    accuracy = total_num_correct_guess / total_num_tokens
    num_token_keys = sorted([key for key in all_metrics.keys() if key.startswith('num_tokens')])
    final_metrics = {}
    for key in num_token_keys:
      modal_name = key.split('_')[-1]
      final_metrics[f'acc_{modal_name}'] = sum(all_metrics[f"num_correct_{modal_name}"]) / sum(all_metrics[f'num_tokens_{modal_name}'])
      final_metrics[f'loss_{modal_name}'] = ((torch.tensor(all_metrics[modal_name]) * torch.tensor(all_metrics[f'num_tokens_{modal_name}'])).sum() / sum(all_metrics[f'num_tokens_{modal_name}'])).item()
      final_metrics[f'num_tokens_{modal_name}'] = sum(all_metrics[f'num_tokens_{modal_name}'])
    final_metrics['loss'] = avg_loss
    final_metrics['accuracy'] = accuracy
    final_metrics['total_num_tokens'] = total_num_tokens
    final_metrics['total_num_correct_guess'] = total_num_correct_guess
    return avg_loss, final_metrics

  def run_validate(self, iter):
    self.model.eval()
    validation_loss, validation_metrics = self.validate()
    validation_metrics['loss'] = validation_loss
    valid_score_counter = None
    amt_score_counter = None
    
    if self.valid_set.modal_direction in ['omr', 'bi'] and 'lmx' in self.valid_set.out_modal_type:
      if self.config.nn_params.compile:
        self.uncompile_model()
      try:
        valid_score_dict, num_samples = self.validate_lmx(num_samples=200)
      except Exception as e:
        print(f"Error in validate_lmx: {e}")
        num_samples = 1
        valid_score_dict = {'SER': 0, 'SERnotuplets': 0}
      valid_score_counter = torch.zeros((3,), device=self.device)
      valid_score_counter[0] = valid_score_dict['SER'] * num_samples
      valid_score_counter[1] = valid_score_dict['SERnotuplets'] * num_samples
      valid_score_counter[2] = num_samples
      if torch.distributed.is_initialized():
        torch.distributed.reduce(valid_score_counter, 0)
    
    if self.valid_set.modal_direction in ['amt', 'bi'] and 'midi' in self.valid_set.out_modal_type:
      if self.config.nn_params.compile:
        self.uncompile_model()
      try:
        if "musicnet" in self.valid_set.dataset_names:
          amt_valid_score_dict, num_samples = self.validate_amt('musicnet', num_samples=4)
        elif "musicnet_ogdac" in self.valid_set.dataset_names:
          amt_valid_score_dict, num_samples = self.validate_amt('musicnet_ogdac', num_samples=4)
        else:
          raise ValueError(f"Invalid dataset name: {self.valid_set.dataset_names}")
      except Exception as e:
        print(f"Error in validate_amt: {e}")
        num_samples = 1
        amt_valid_score_dict = {'onset_f': 0, 'offset_f': 0}
      amt_score_counter = torch.zeros((3,), device=self.device)
      amt_score_counter[0] = amt_valid_score_dict['onset_f'] * num_samples
      amt_score_counter[1] = amt_valid_score_dict['offset_f'] * num_samples
      amt_score_counter[2] = num_samples
      if torch.distributed.is_initialized():
        torch.distributed.reduce(amt_score_counter, 0)
      
    if torch.distributed.is_initialized():
      counter = torch.zeros((len(self.modal_pairs_code) * 3 + 3,), device=self.device)
      for key, value in validation_metrics.items():
        if key.startswith('acc_'):
          counter[self.modal_pairs_map[key.split('_')[-1]]] = value * validation_metrics['num_tokens_'+key.split('_')[-1]]
        elif key.startswith('loss_'):
          counter[self.modal_pairs_map[key.split('_')[-1]] + len(self.modal_pairs_code)] = value * validation_metrics['num_tokens_'+key.split('_')[-1]]
        elif key.startswith('num_tokens_'):
          counter[self.modal_pairs_map[key.split('_')[-1]] + len(self.modal_pairs_code) * 2] = value
      counter[len(self.modal_pairs_code) * 3] = validation_loss * validation_metrics['total_num_tokens']
      counter[len(self.modal_pairs_code) * 3 + 1] = validation_metrics['total_num_correct_guess']
      counter[len(self.modal_pairs_code) * 3 + 2] = validation_metrics['total_num_tokens']
      torch.distributed.reduce(counter, 0)
    

    if self.make_log:
      if torch.distributed.is_initialized():
        for i, modal_pair in enumerate(self.modal_pairs_code):
          validation_metrics[f'acc_{modal_pair}'] = (counter[i] / counter[i + len(self.modal_pairs_code) * 2]).item()
          validation_metrics[f'loss_{modal_pair}'] = (counter[i + len(self.modal_pairs_code)] / counter[i + len(self.modal_pairs_code) * 2]).item()
        validation_metrics['loss'] = (counter[len(self.modal_pairs_code) * 3] / counter[len(self.modal_pairs_code) * 3 + 2]).item()
        validation_metrics['accuracy'] = (counter[len(self.modal_pairs_code) * 3 + 1] / counter[len(self.modal_pairs_code) * 3 + 2]).item()
      if valid_score_counter is not None:
        validation_metrics['SER'] = valid_score_counter[0] / valid_score_counter[2]
        validation_metrics['SERnotuplets'] = valid_score_counter[1] / valid_score_counter[2]
      if amt_score_counter is not None:
        validation_metrics['onset_f'] = amt_score_counter[0] / amt_score_counter[2]
        validation_metrics['offset_f'] = amt_score_counter[1] / amt_score_counter[2]
        
      del_keys = []
      for key, value in validation_metrics.items():
        if key.startswith('num_tokens_'):
          del_keys.append(key)
        elif math.isnan(value):
          del_keys.append(key)
      for key in del_keys:
        validation_metrics.pop(key)
      validation_metrics = self._rename_dict(validation_metrics, 'valid')    
      self.wandb_run.log(validation_metrics, step=iter)
    if self.config.nn_params.compile:
      self.compile_model()
    return validation_loss

  @torch.inference_mode()
  def validate_lmx(self, num_samples=1000):
    
    valid_loader = self.get_specific_validset_loader('olimpic', 'lmx', num_samples=num_samples)
    model = self.model.module if self.use_ddp else self.model
    ser_dict = Evaluator.eval_omr_loader(model, valid_loader)
    return ser_dict, len(valid_loader.dataset)
  
  @torch.inference_mode()
  def validate_amt(self, dataset_name='musicnet', num_samples=10):
    model = self.model.module if self.use_ddp else self.model
    f1_dict, num_samples = Evaluator.eval_amt_dataset(model, self.valid_set, self.decoder, dataset_name, batch_size=self.batch_size, num_samples=num_samples, max_len_sec=60, rank=self.gpu_id, world_size=self.world_size)
    return f1_dict, num_samples

  def get_specific_validset_loader(self, dataset_name, out_modal, num_samples=None):
    path_pairs = {dataset_name: [path_pair for (name, path_pair) in self.valid_set.path_pairs if name == dataset_name]}
    if num_samples is not None:
      path_pairs[dataset_name] = path_pairs[dataset_name][:num_samples]
    selected_dataset = MultimodalTokenDataset(path_pairs, 
                                              self.valid_set.max_length, 
                                              (self.valid_set.in_idx_handler, self.valid_set.out_idx_handler), 
                                              self.valid_set.in_modal_type, 
                                              [out_modal],
                                              preload_data=False,
                                              is_valid=True)
    
    if self.use_ddp:
      sampler = DistributedEvalSampler(selected_dataset,
                                        num_replicas=self.world_size,
                                        rank=self.gpu_id,
                                        shuffle=False,
                                        seed=self.config.general.seed,
                                        )
    else:
      sampler = None
    multiprocessing_context = torch.multiprocessing.get_context('fork') if self.num_workers > 0 else None
    prefetch_factor = 2 if self.num_workers > 0 else None
    return DataLoader(selected_dataset, 
                      batch_size=self.batch_size, 
                      shuffle=False, 
                      drop_last=False, 
                      collate_fn=self.collate_fn, 
                      num_workers=self.num_workers,
                      multiprocessing_context=multiprocessing_context,
                      prefetch_factor=prefetch_factor,
                      pin_memory=True,
                      sampler=sampler)

  @torch.inference_mode()
  def inference(self, audio_path):
    return
  
  @torch.inference_mode()
  def inference_and_log(self, n_iter, num_inference=1):
    if self.config.nn_params.compile:
      self.uncompile_model()
    self.model.eval()
    
    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
      model = self.model.module
    else:
      model = self.model

    sampling_method = self.config.inference_params.sampling.method
    sampling_threshold = self.config.inference_params.sampling.threshold
    sampling_temperature = self.config.inference_params.sampling.temperature

    if torch.distributed.is_initialized():
      rank = torch.distributed.get_rank()
      world_size = torch.distributed.get_world_size()
    else:
      rank = 0
      world_size = 1
      
    log_dict = None
    try:
      for i in range(num_inference):
        # Get data from each dataset among {'lsyt', 'slakh', 'grandstaff-lmx', 'olimpic', 'maestro', 'musicnet'}
        # 2 for in/out modality switch: odd idx and even idx have switched modality in-n-out
        batch = []
        dataset_names = []
        for dataset_name, start_idx in self.valid_set.dataset_start_indices.items():
          indices = list(range(start_idx + i*world_size*2, start_idx + (i+1)*world_size*2 + 100))
          indices = [j for j in indices if j % world_size == rank]
          batch.append(self.valid_set[indices[0]]) 
          batch.append(self.valid_set[indices[1]])
          dataset_names.append(dataset_name)
          dataset_names.append(dataset_name)
        batch = self.collate_fn(batch)

        in_modal, in_mask, target_in, target_out, modal_idx, token_heights, in_pos, _ = batch['in_modal'], batch['in_mask'], batch['target_in'], batch['target_out'], batch['modal_idx'], batch['token_height'], batch['in_pos'], batch['target_in_pos']
        
        in_modal = in_modal.to(self.device)
        in_mask = in_mask.to(self.device)
        modal_idx = modal_idx.to(self.device)
        token_heights = token_heights.to(self.device)
        in_pos = in_pos.to(self.device)
        
        start_time = time.time()      

        print(f"Inference: {n_iter}th iter: data {i}: {sampling_method}_{sampling_threshold}_{sampling_temperature}")

        inferenced_output = model.inference(
          in_modal=in_modal,
          in_pos=in_pos,
          modal_idx=modal_idx,
          in_mask=in_mask,
          token_heights=token_heights,
          sampling_method=sampling_method, 
          threshold=sampling_threshold, 
          temperature=sampling_temperature, 
          manual_seed=i
        )
        
        # decoder_hook.remove()
        
        if len(inferenced_output) == 0:
          continue
        
        print(f"Inference: {n_iter}th iter: data {i}: Time: {time.time() - start_time:.4f}")
        print(f"Inference: {n_iter}th iter: data {i}: Len: {inferenced_output.shape[1]}")
        if (n_iter+1) % (self.num_cycles_for_inference * self.iterations_per_validation_cycle) == 0:
          input_decoded_file_fns = self.decoder(batch['in_modal'], batch['modal_idx'][:,0], dataset_names, 'input', token_heights=token_heights, use_in=True, n_iter=n_iter)
          output_decoded_file_fns = self.decoder(batch['target_out'], batch['modal_idx'][:,1], dataset_names, 'target', token_heights=token_heights, use_in=False, n_iter=n_iter)
          log_dict = self.decoder.make_log_dict(batch['modal_idx'], dataset_names, input_decoded_file_fns, 'input', log_dict=log_dict, n_iter=n_iter)
          log_dict = self.decoder.make_log_dict(batch['modal_idx'], dataset_names, output_decoded_file_fns, 'target', log_dict=log_dict, n_iter=n_iter)
        
        pred_decoded_file_fns = self.decoder(inferenced_output, batch['modal_idx'][:,1], dataset_names, 'prediction', token_heights=token_heights, use_in=False, n_iter=n_iter)
        log_dict = self.decoder.make_log_dict(batch['modal_idx'], dataset_names, pred_decoded_file_fns, 'prediction', log_dict=log_dict, n_iter=n_iter)
      if self.make_log and rank == 0:
        self.wandb_run.log(log_dict, step=(n_iter))
        print(f"Inference: {n_iter}th iter: Log: Done")
    
    except Exception as e:
      print(f"Inference: {n_iter}th iter: Error: {e}")
    if self.config.nn_params.compile:
      self.compile_model()
