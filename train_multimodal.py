import os
import copy
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from umust.train_utils import CosineLRScheduler, CosineAnnealingWarmUpRestarts, MultimodalLoss
from umust.constants import *
from umust import trainer
from umust.utils import *

import sys

get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def ddp_setup(rank, world_size, backend='nccl', port=12355):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  # Autoregressive inference/decode cycles (general.infer_and_log) can hold a
  # rank far past NCCL's default collective timeout; use a generous limit.
  init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(hours=4))
  torch.cuda.set_device(rank)

def generate_experiment_name(config):
  # add base hyperparameters to the experiment name
  model_dim = config.nn_params.model_dim
  main_dropout = config.nn_params.dropout_shared
  batch_size = config.train_params.batch_size
  lr_decay_rate = config.train_params.decay_step_rate
  memo = config.general.memo

  if config.nn_params.type == 'multimodal_trans':
    num_enc_layers = config.nn_params.encoder_params.num_layer
    num_dec_layers = config.nn_params.decoder_params.decoder.num_layer
    experiment_name = f"{get_ts()}:{config.data.exp_name}:{config.nn_params.type}:dim-{model_dim}:nLayers-{num_enc_layers}_{num_dec_layers}:dropout-{main_dropout}:batchSize-{batch_size}:lrDecay-{lr_decay_rate}:{memo}"
  else:
    raise ValueError(f"Invalid nn_params.type: {config.nn_params.type}")

  return experiment_name

def setup_log(config):
  if config.general.resume and config.general.make_log:
    experiment_name = config.general.resume
    wandb_config = config.wandb_config
    id = str(list((Path('wandb') / experiment_name).glob('*.wandb'))[0].stem).replace('run-', '')
    wandb_run = wandb.init(
      project=wandb_config.project,
      entity=wandb_config.entity,
      name=experiment_name,
      config = OmegaConf.to_container(config),
      resume= True,
      id=id
    )
    save_dir = wandb_run.dir + '/checkpoints/'

  else:
    if hasattr(config, 'general.exp_name') and config.general.exp_name != '':
      experiment_name = config.general.exp_name
    else:
      experiment_name = generate_experiment_name(config)
    if config.general.make_log:
      wandb_config = config.wandb_config

      wandb_run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=experiment_name,
        config = OmegaConf.to_container(config)
      )

      save_dir = wandb_run.dir + '/checkpoints/'
      Path(save_dir).mkdir(exist_ok=True, parents=True)

      (Path('./wandb') / experiment_name).symlink_to(Path(wandb_run.dir).parent, target_is_directory=True)

    else:
      wandb_run = None
      save_dir = f'wandb/debug/{experiment_name}/files/checkpoints/'
      Path(save_dir).mkdir(exist_ok=True, parents=True)

  OmegaConf.save(config=config, f=str(Path(save_dir).parent / 'config.yaml'))
  return wandb_run, save_dir

def prepare_trainer(config, wandb_run, save_dir, rank):
  nn_params = config.nn_params
  data_config = config.data

  if not hasattr(data_config, 'max_pt_x_len'): # For old configs
    print("max_pt_x_len is not set in data_config. Setting to max_seq_len['pt']")
    with open_dict(data_config):
      data_config.max_pt_x_len = data_config.max_seq_len['pt']
  if not hasattr(data_config, 'out_pt_height_token'): # For old configs
    print("out_pt_height_token is not set in data_config. Setting to False")
    with open_dict(data_config):
      data_config.out_pt_height_token = False

  if config.finetune_params.finetune:
    prev_config = load_config(Path(config.finetune_params.finetune_path) / 'config.yaml')
    try:
      prev_config = convert_wandb_style_config_to_omega_config(prev_config)
    except:
      pass
    data_config.in_modal_type = prev_config.data.in_modal_type
    data_config.out_modal_type = prev_config.data.out_modal_type
    data_config.modal_direction = prev_config.data.modal_direction
    data_config.image_height = prev_config.data.image_height
    data_config.max_seq_len = prev_config.data.max_seq_len
    if not hasattr(prev_config.data, 'max_pt_x_len'): # For old configs
      with open_dict(prev_config):
        prev_config.data.max_pt_x_len = prev_config.data.max_seq_len['pt']
    data_config.max_pt_x_len = prev_config.data.max_pt_x_len
    data_config.lmx_vocab_path = prev_config.data.lmx_vocab_path
    data_config.midi_max_shift = prev_config.data.midi_max_shift
    if not hasattr(prev_config.data, 'out_pt_height_token'):
      with open_dict(prev_config):
        prev_config.data.out_pt_height_token = False
    data_config.out_pt_height_token = prev_config.data.out_pt_height_token
    config.data = data_config
    OmegaConf.save(config=config, f=str(Path(save_dir).parent / 'config.yaml')) 

  
  vq_model = None
  vq_emb = None
  dac_model = None
  dac_emb = None
  if data_config.vq_model:
    vq_model, vq_emb = get_vq_model(config)
    vq_model.cpu()
  if data_config.dac_model:
    dac_model, dac_emb = get_dac_model(config)
    dac_model.cpu()
  
  dataset = get_dataset(config)
  
  trainset, validset, testset = dataset.get_datasets()
  
  model = get_model(config, vq_emb, dac_emb, dataset)

  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total Num Params: {total_params}")
  
  # log in wandb
  if config.general.make_log:
    if config.general.resume is None:
      wandb_run.log({'nn_total_params': total_params})

  loss_fn = MultimodalLoss(dataset.in_idx_handler, dataset.out_idx_handler)

  optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_params.initial_lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
  
  scheduler_dict = {
    'not-using': None,
    'cosineannealingwarmuprestarts': CosineAnnealingWarmUpRestarts, 
    'cosinelr': CosineLRScheduler,
  }
  
  if config.train_params.min_lr is not None:
    eta_min = config.train_params.min_lr
  else:
    eta_min = 0

  if config.train_params.min_lr_ratio is not None:
    lr_min_ratio = config.train_params.min_lr_ratio
  else:
    lr_min_ratio = 0.1

  if scheduler_dict[config.train_params.scheduler] == CosineAnnealingWarmUpRestarts:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, T_0=config.train_params.num_steps_per_cycle, T_mult=2, eta_min=eta_min, eta_max=config.train_params.max_lr,  T_up=config.train_params.warmup_steps , gamma=config.train_params.gamma)
  
  elif scheduler_dict[config.train_params.scheduler] == CosineLRScheduler:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, total_steps=config.train_params.num_iter * config.train_params.decay_step_rate, warmup_steps=config.train_params.warmup_steps, lr_min_ratio=lr_min_ratio, cycle_length=1.0)
  
  else:
    scheduler = None
  
  iter = 0
  if config.general.resume:
    ckpt_dir = Path('wandb') / Path(config.general.resume) / 'files' / 'checkpoints'
    ckpt_path = ckpt_dir.glob('*.pt')
    ckpt_path = list(ckpt_path)
    if ckpt_path:
      print(f"Loading checkpoint from {config.general.resume}")
      if (ckpt_dir / 'last_checkpoint.pt').exists():
        last_ckpt = torch.load(ckpt_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=True)
      else:
        last_ckpt = torch.load(max(ckpt_path, key=lambda p: int(p.stem.split('_')[0][4:])), map_location='cpu', weights_only=True)
      # ckpt_path = [x for x in ckpt_path if x.stem.startswith('iter279999')][0]
      # last_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

      # remove module. prefix for ddp
      keys_to_modify = [k for k in last_ckpt['model_state_dict'].keys() if k.startswith('module.')]
      for k in keys_to_modify:
        v = last_ckpt['model_state_dict'][k]
        last_ckpt['model_state_dict'][k[7:]] = v
        del last_ckpt['model_state_dict'][k]
      
      try:
        model.load_state_dict(last_ckpt['model_state_dict'])
      except:
        model.compile_model()
        model.load_state_dict(last_ckpt['model_state_dict'])
        model.uncompile_model()
      model = model.to(f"cuda:{rank}")
      optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_params.initial_lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
      if scheduler_dict[config.train_params.scheduler] == CosineAnnealingWarmUpRestarts:
        scheduler = scheduler_dict[config.train_params.scheduler](optimizer, T_0=config.train_params.num_steps_per_cycle, T_mult=2, eta_min=eta_min, eta_max=config.train_params.max_lr,  T_up=config.train_params.warmup_steps , gamma=config.train_params.gamma)
      
      elif scheduler_dict[config.train_params.scheduler] == CosineLRScheduler:
        scheduler = scheduler_dict[config.train_params.scheduler](optimizer, total_steps=config.train_params.num_iter * config.train_params.decay_step_rate, warmup_steps=config.train_params.warmup_steps, lr_min_ratio=lr_min_ratio, cycle_length=1.0)
      
      else:
        scheduler = None
      optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
      if scheduler is not None:
        scheduler.load_state_dict(last_ckpt['scheduler_state_dict'])
      iter = last_ckpt['iter']+1
      print(f"Loaded checkpoint from {config.general.resume}")
      del last_ckpt
      
  
  bucket_config = data_config.bucket
  training_instance = trainer.MultimodalTrainer(model, 
                                                optimizer, 
                                                scheduler, 
                                                loss_fn, 
                                                trainset, 
                                                validset, 
                                                save_dir, 
                                                use_ddp=config.use_ddp, 
                                                bucket_config=bucket_config, 
                                                use_fp16=config.use_fp16, 
                                                world_size=config.train_params.world_size, 
                                                batch_size=config.train_params.batch_size, 
                                                gpu_id=rank, 
                                                config=config, 
                                                dac_model=dac_model,
                                                rq_model=vq_model,
                                                wandb_run=wandb_run, 
                                                num_workers=config.train_params.num_workers,
                                                start_iter=iter)
  return training_instance


def run_train_exp(rank, config, world_size:int=1):
  
  import torch._dynamo
  torch._dynamo.config.suppress_errors = True
  
  if config.use_ddp: 
    ddp_setup(rank, world_size, port=config.general.ddp_port)
  
  config = copy.deepcopy(config)
  config.train_params.world_size = world_size
  
  if rank != 0:
    config.general.make_log = False
    config.general.infer_and_log = False

  wandb_run, save_dir = setup_log(config)
  
  if config.general.seed is not None:
    set_seed(config.general.seed)
    
  training_module = prepare_trainer(config, wandb_run, save_dir, rank)
  if config.finetune_params.finetune:
    training_module.train_by_num_iter(config.finetune_params.iter)
  else:
    training_module.train_by_num_iter(config.train_params.num_iter)

  if config.use_ddp:
    destroy_process_group()


@hydra.main(version_base=None, config_path="./config/", config_name="config_mm")
def main(config: DictConfig):
  os.environ['QT_QPA_PLATFORM'] = 'offscreen'
  if config.use_ddp:
    # world_size = torch.cuda.device_count()
    world_size = config.train_params.world_size
    print(f"world_size: {world_size}")
    mp.spawn(run_train_exp, args=(config, world_size), nprocs=world_size)
    
  else:
    run_train_exp(0, config) # single gpu




if __name__ == '__main__':
  main()