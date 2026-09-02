import os

import torch
import numpy as np
import random

from pathlib import Path
from omegaconf import OmegaConf, open_dict

from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model

from umust.dac_utils import LSDAC

from umust import model_zoo, data_utils
from umust.constants import *
from umust.weight_transfer_util import transfer_weight_from_whisper


def load_vq_model(config:OmegaConf):
  """
  Loads a vector quantization (VQ) model based on the provided configuration.
  Args:
    config (OmegaConf): Configuration object containing model parameters.
  Returns:
    torch.nn.Module: The loaded VQ model.
  Raises:
    ValueError: If the nn_params.type in the configuration is invalid.
  The function supports different types of VQ models specified by `config.nn_params.type`:
    - 'lmx2rvq' or 'rvq': Loads a model with shared or unshared codebooks.
  The configuration object should contain the following attributes:
    - config.data.image.vq_model (str): Name of the VQ model.
    - config.data.image.vq_model_fit (str): Fit type of the VQ model.
    - config.data.image.compress_factor (int): Compression factor for the image.
    - config.data.image.vocab.codebook_size (int): Size of the codebook.
    - config.data.image.height (int): Height of the image.
    - config.data.image.tags (list): List of tags associated with the image.
    - config.nn_params.type (str): Type of the neural network parameters.
    - config.data.image.vocab.n_codebook (int, optional): Number of codebooks (for 'lmx2rvq' or 'rvq' types).
    - config.data.image.vocab.shared_codebook (bool, optional): Whether the codebook is shared (for 'lmx2rvq' or 'rvq' types).
  The function constructs the model string based on the configuration and loads the corresponding model and checkpoint.
  """
  vq_model_name = config.data.image.vq_model
  vq_model_fit = config.data.image.vq_model_fit
  image_compress_factor = config.data.image.compress_factor
  codebook_size = config.data.image.vocab.codebook_size
  image_height = config.data.image.height
  image_tags = "_".join(config.data.image.tags)
  if config.nn_params.type in ['lmx2rvq', 'rvq']:
    n_codebook = config.data.image.vocab.n_codebook
    rvq_codebook_shared = "shared" if config.data.image.vocab.shared_codebook else "unshared"
    model_string = f"{vq_model_name}_f{image_compress_factor}_c{codebook_size}_k{n_codebook}_{rvq_codebook_shared}_{image_height}p_{image_tags}_{vq_model_fit}"
    vq_model_dir = Path(config.data.get("vq_model_dir", "vq_models")) / model_string
    if not vq_model_dir.exists():
      raise FileNotFoundError(
        f"Image tokenizer checkpoint not found: {vq_model_dir}. "
        f"Place the pretrained RQ-VAE checkpoint (config.yaml + *.pt) under this directory, "
        f"or set data.vq_model_dir to its parent directory."
      )
    vq_config_path = list(vq_model_dir.rglob("config.yaml"))[0]
    vq_config = load_config(vq_config_path)
    vq_config.arch = augment_arch_defaults(vq_config.arch)
    
    vq_model, _ = create_model(vq_config.arch)
    
    ckpt_path = list(vq_model_dir.rglob("*.pt"))[0]
    
    vq_model.load_state_dict(torch.load(ckpt_path)["state_dict"])
  else:
    raise ValueError(f"Invalid nn_params.type: {config.nn_params.type}")
  return vq_model

def load_vq_model_mm(config:OmegaConf):
  vq_model_name = config.data.vq_model
  image_compress_factor = config.data.image_compress_factor
  codebook_size = config.data.codebook_size
  n_codebook = config.data.n_codebook
  model_string = f"{vq_model_name}_f{image_compress_factor}_c{codebook_size}_k{n_codebook}"
  vq_model_dir = Path(config.data.get("vq_model_dir", "vq_models")) / model_string
  if not vq_model_dir.exists():
    raise FileNotFoundError(
      f"Image tokenizer checkpoint not found: {vq_model_dir}. "
      f"Place the pretrained RQ-VAE checkpoint (config.yaml + *.pt) under this directory, "
      f"or set data.vq_model_dir to its parent directory."
    )
  vq_config_path = list(vq_model_dir.rglob("config.yaml"))[0]
  vq_config = load_config(vq_config_path)
  vq_config.arch = augment_arch_defaults(vq_config.arch)
  
  vq_model, _ = create_model(vq_config.arch)
  
  ckpt_path = list(vq_model_dir.rglob("*.pt"))[0]

  vq_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"])
  return vq_model

def get_vq_model(config):
  nn_params = config.nn_params
  encoder_params = config.nn_params.encoder_params
  decoder_params = config.nn_params.decoder_params
  data_config = config.data

  vq_model = None
  vq_emb = None

  if nn_params.type in ['vq', 'rvq', 'lmx2vq', 'lmx2rvq']:
    vq_model = load_vq_model(config)
    if decoder_params.use_vq_emb:
      vq_emb = vq_model.quantize.embedding.weight.clone()
      assert vq_emb.shape[0] == data_config.image.vocab.codebook_size, f"VQ embedding n_codebook size mismatch: vq's n_codebook {vq_emb.shape[0]} != vocab.n_codebook {data_config.image.vocab.codebook_size}"
  elif nn_params.type == 'multimodal_trans':
    vq_model = load_vq_model_mm(config)
    if nn_params.use_vq_emb:
      vq_emb = {}
      for i in range(data_config.n_codebook):
        vq_emb[i] = vq_model.quantizer.codebooks[i].weight.clone()[:data_config.codebook_size].detach()

  return vq_model, vq_emb

def get_fluidsynth():
  """midi2audio only looks for ~/.fluidsynth/default_sound_font.sf2; fall back
  to the common system locations of the GM soundfont (fluid-soundfont-gm)."""
  from midi2audio import FluidSynth, DEFAULT_SOUND_FONT
  if os.path.exists(os.path.expanduser(DEFAULT_SOUND_FONT)):
    return FluidSynth()
  for candidate in (
    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
    '/usr/share/soundfonts/FluidR3_GM.sf2',
    '/usr/share/sounds/sf2/default-GM.sf2',
  ):
    if os.path.exists(candidate):
      return FluidSynth(sound_font=candidate)
  return FluidSynth()

def get_dac_model(config):
  data_config = config.data
  dac_model_dir = Path(data_config.get("dac_model_dir", "dac_models")) / data_config.dac_model

  if not (dac_model_dir / 'weights.pth').exists():
    raise FileNotFoundError(
      f"Audio tokenizer checkpoint not found: {dac_model_dir / 'weights.pth'}. "
      f"Place the pretrained DAC checkpoint (weights.pth) under this directory, "
      f"or set data.dac_model_dir to its parent directory."
    )

  dac_model = LSDAC.load( dac_model_dir / 'weights.pth' )
  dac_emb = None

  if config.nn_params.use_dac_emb:
    dac_emb = {}
    for i in range(data_config.n_codebook):
      weight = dac_model.quantizer.quantizers[i].out_proj.weight.clone()
      bias = dac_model.quantizer.quantizers[i].out_proj.bias.clone()
      codebook = dac_model.quantizer.quantizers[i].codebook.weight.clone()
    
      embedding = weight.squeeze(-1) @ codebook.T
      embedding = embedding.T + bias
      dac_emb[i] = embedding.detach()
    
  return dac_model, dac_emb

def get_dataset(config):
  nn_params = config.nn_params
  encoder_params = config.nn_params.encoder_params
  decoder_params = config.nn_params.decoder_params
  data_config = config.data

  # get dataset maker class by name
  # { OnTheFlyAudioVQDatasetMaker, OnMemoryAudioVQDatasetMaker }
  if nn_params.type == 'vq':
    dataset = getattr(data_utils, data_config.dataset)( 
      data_path = Path.home() / 'userdata' / data_config.data_path,
      metadata_dir = Path.cwd() / 'metadata' / data_config.metadata_dir,
      image_height = data_config.image.height,
      image_compress_factor = data_config.image.compress_factor,
      image_tags = data_config.image.tags,
      audio_channels = data_config.audio.num_channels, # 'mono' or 'stereo'
      audio_sr = data_config.audio.sample_rate,
      resample = data_config.audio.resample,
      vq_model = data_config.image.vq_model, # 'sqocremavq'
      vq_model_fit = data_config.image.vq_model_fit, # 'fit' or 'unfit
      shifted = data_config.image.shifted,
      codebook_size = data_config.image.vocab.codebook_size, # 16384
      image_token_max_width = decoder_params.max_tok_width, # 150toks
      num_special_tokens = data_config.image.vocab.num_special_tokens,
      audio_length_threshold = data_config.audio.length_threshold, # 76s
      image_width_threshold = data_config.image.width_threshold, # 2000px
    ) 
  elif nn_params.type == 'rvq':
    dataset = getattr(data_utils, data_config.dataset)(
      data_path = Path.home() / 'userdata' / data_config.data_path,
      metadata_dir = Path.cwd() / 'metadata' / data_config.metadata_dir,
      image_height = data_config.image.height,
      image_compress_factor = data_config.image.compress_factor,
      image_tags = data_config.image.tags,
      audio_channels = data_config.audio.num_channels, # 'mono' or 'stereo'
      audio_sr = data_config.audio.sample_rate,
      resample = data_config.audio.resample,
      vq_model = data_config.image.vq_model, # 'sqocremavq'
      vq_model_fit = data_config.image.vq_model_fit, # 'fit' or 'unfit
      rvq_codebook_shared = data_config.image.vocab.shared_codebook,
      shifted = data_config.image.shifted,
      codebook_size = data_config.image.vocab.codebook_size, # 16384
      n_codebook = data_config.image.vocab.n_codebook,
      image_token_max_width = decoder_params.max_tok_width, # 150toks
      num_special_tokens = data_config.image.vocab.num_special_tokens,
      audio_length_threshold = data_config.audio.length_threshold, # 76s
      image_width_threshold = data_config.image.width_threshold, # 2000px
    )
  elif nn_params.type == 'pianoroll':
    dataset = getattr(data_utils, data_config.dataset)( 
      data_path = Path.home() / 'userdata' / data_config.data_path,
      metadata_dir = Path.cwd() / 'metadata' / data_config.metadata_dir,
      audio_channels = data_config.audio.num_channels, # 'mono' or 'stereo'
      audio_sr = data_config.audio.sample_rate,
      resample = data_config.audio.resample,
      audio_length_min = data_config.audio.length_min,
      audio_length_max = data_config.audio.length_max,
    )
  elif nn_params.type == 'lmx':
    dataset = getattr(data_utils, data_config.dataset)(
      data_path = Path.home() / 'userdata' / data_config.data_path,
      metadata_dir = Path.cwd() / 'metadata' / data_config.metadata_dir,
      audio_channels = data_config.audio.num_channels, # 'mono' or 'stereo'
      audio_sr = data_config.audio.sample_rate,
      max_seq_len = decoder_params.max_tok_width,
      resample = data_config.audio.resample,
      audio_length_min = data_config.audio.length_min,
      audio_length_max = data_config.audio.length_max,
      lmx_length_max = data_config.lmx.length_max,
      num_special_tokens = data_config.lmx.vocab.num_special_tokens,
    )
  elif nn_params.type == 'lmx2vq':
    dataset = getattr(data_utils, data_config.dataset)(
      data_path = Path.home() / 'userdata' / data_config.data_path,
      metadata_dir = Path.cwd() / 'metadata' / data_config.metadata_dir,
      image_height = data_config.image.height,
      image_compress_factor = data_config.image.compress_factor,
      image_tags = data_config.image.tags,
      vq_model = data_config.image.vq_model,
      vq_model_fit = data_config.image.vq_model_fit,
      shifted = data_config.image.shifted,
      codebook_size = data_config.image.vocab.codebook_size,
      max_lmx_seq_len = encoder_params.max_lmx_length,
      image_token_max_width = decoder_params.max_tok_width,
      num_special_tokens = data_config.lmx.vocab.num_special_tokens,
      lmx_length_max = data_config.lmx.length_max,
      image_width_threshold = data_config.image.width_threshold,
    )
  elif nn_params.type == 'lmx2rvq':
    dataset = getattr(data_utils, data_config.dataset)(
      data_path = Path.home() / 'userdata' / data_config.data_path,
      metadata_dir = Path.cwd() / 'metadata' / data_config.metadata_dir,
      image_height = data_config.image.height,
      image_compress_factor = data_config.image.compress_factor,
      image_tags = data_config.image.tags,
      vq_model = data_config.image.vq_model,
      vq_model_fit = data_config.image.vq_model_fit,
      n_codebook = data_config.image.vocab.n_codebook,
      rvq_codebook_shared = data_config.image.vocab.shared_codebook,
      shifted = data_config.image.shifted,
      codebook_size = data_config.image.vocab.codebook_size,
      max_lmx_seq_len = encoder_params.max_lmx_length,
      image_token_max_width = decoder_params.max_tok_width,
      num_special_tokens = data_config.lmx.vocab.num_special_tokens,
      lmx_length_max = data_config.lmx.length_max,
      image_width_threshold = data_config.image.width_threshold,
    )
  elif nn_params.type == 'multimodal_trans':
    if not hasattr(data_config, 'num_measure_to_slice'):
      print("num_measure_to_slice is not set in data_config. Setting to 4")
      with open_dict(data_config):
        data_config.num_measure_to_slice = 4
    if not hasattr(data_config, 'lmx_vocab_path'):
      print("lmx_vocab_path is not set in data_config. Setting to vocab/lmx_vocab_singletoken_asap.txt")
      with open_dict(data_config):
        data_config.lmx_vocab_path = 'vocab/lmx_vocab_singletoken_asap.txt'
    if not hasattr(data_config, 'midi_slice_len'):
      print("midi_slice_len is not set in data_config. Setting to 10.0")
      with open_dict(data_config):
        data_config.midi_slice_len = 10.0
    if not hasattr(config.train_params, 'num_workers'):
      print("num_workers is not set in config.train_params. Setting to 4")
      with open_dict(config.train_params):
        config.train_params.num_workers = 4
    if not hasattr(data_config, 'tps'):
      print("tps is not set in data_config. Setting to 100")
      with open_dict(data_config):
        data_config.tps = 100
    if not hasattr(data_config, 'max_pt_x_len'): # For old configs
      print("max_pt_x_len is not set in data_config. Setting to max_seq_len['pt']")
      with open_dict(data_config):
        data_config.max_pt_x_len = data_config.max_seq_len['pt']
    if not hasattr(data_config, 'out_pt_height_token'): # For old configs
      print("out_pt_height_token is not set in data_config. Setting to False")
      with open_dict(data_config):
        data_config.out_pt_height_token = False

    dataset = getattr(data_utils, data_config.dataset)(
      data_path = data_config.data_path,
      data_dir = data_config.data_dir,
      metadata_dir = Path.cwd() / data_config.metadata_dir,
      n_codebook = data_config.n_codebook,
      codebook_size = data_config.codebook_size,
      max_seq_len = data_config.max_seq_len,
      max_pt_x_len = data_config.max_pt_x_len,
      num_special_tokens = data_config.num_special_tokens,
      image_height = data_config.image_height, # This must be the max image height in the dataset
      image_compress_factor = data_config.image_compress_factor,
      midi_max_shift = data_config.midi_max_shift,
      in_modal_type = data_config.in_modal_type,
      out_modal_type = data_config.out_modal_type,
      debug = config.general.debug,
      preload_data = data_config.preload_data,
      modal_direction = data_config.modal_direction,
      num_measure_to_slice = data_config.num_measure_to_slice,
      lmx_vocab_path = data_config.lmx_vocab_path,
      midi_slice_len = data_config.midi_slice_len,
      tps = data_config.tps,
      out_pt_height_token = data_config.out_pt_height_token,
    )
  else:
    raise ValueError(f"Invalid nn_params.type: {nn_params.type}")
  
  return dataset

def normalize_compiled_state_dict(state_dict):
  # torch.compile can be applied to individual submodules, so _orig_mod
  # shows up mid-key at varying depths rather than as a single prefix
  return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

def load_model_state_dict(model, state_dict):
  try:
    model.load_state_dict(state_dict)
  except RuntimeError:
    state_dict = normalize_compiled_state_dict(state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
      raise RuntimeError(
        f"state_dict mismatch after normalizing _orig_mod keys: "
        f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys")
  return model

def get_model(config, vq_emb, dac_emb, dataset):
  nn_params = config.nn_params
  if config.finetune_params.finetune:
    pre_config = load_config(config.finetune_params.finetune_path)
    try:
      pre_config = convert_wandb_style_config_to_omega_config(pre_config)
    except:
      pass
    nn_params = pre_config.nn_params
    ckpt_path = get_last_iter_ckpt_path(config.finetune_params.finetune_path)
  
  if nn_params.type == 'vq' or nn_params.type == 'rvq':
    model = getattr(model_zoo, nn_params.model_name)(
      nn_params=nn_params,
      vocab=dataset.vocab,
      vq_emb=vq_emb,
    )
  elif nn_params.type == 'pianoroll':
    model = getattr(model_zoo, nn_params.model_name)(
      nn_params=nn_params,
      output_features=MAX_MIDI-MIN_MIDI+1,
    )
  elif nn_params.type == 'lmx':
    model = getattr(model_zoo, nn_params.model_name)(
      nn_params=nn_params,
      vocab=dataset.vocab,
    )
  elif nn_params.type == 'lmx2vq' or nn_params.type == 'lmx2rvq':
    model = getattr(model_zoo, nn_params.model_name)(
      nn_params=nn_params,
      lmx_vocab=dataset.lmx_vocab,
      vq_vocab=dataset.vq_vocab,
      vq_emb=vq_emb,
    )
  elif nn_params.type == 'multimodal_trans':
    model = model_zoo.MultimodalTranslator(
      nn_params=nn_params,
      in_vocab=dataset.in_idx_handler,
      out_vocab=dataset.out_idx_handler,
      vq_emb=vq_emb,
      dac_emb=dac_emb,
    )
  else:
    raise ValueError(f"Invalid nn_params.type: {nn_params.type}")
  
  if hasattr(config, 'pretrained') and config.pretrained.use_pretrained:
    if 'whisper' in config.pretrained.model:
      weight = torch.load(config.pretrained.path, map_location='cpu', weights_only=False)
      model = transfer_weight_from_whisper(model, weight)
      print(f"Loaded pretrained weight from {config.pretrained.path}")
    # elif 'T5' in config.pretrained.model:
    #   weight = torch.load(config.pretrained.path)
    #   model = transfer_weight_from_T5(model, weight)
    #   print(f"Loaded pretrained weight from {config.pretrained.path}")
    else:
      raise ValueError(f"Invalid pretrained model: {config.pretrained.model}")  
  
  if config.finetune_params.finetune:
    print(f"Loading pretrained model from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    load_model_state_dict(model, state_dict)
  return model




def set_seed(seed):
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def load_config(config_path):
  config_path = Path(config_path)
  if config_path.is_file():
    config = OmegaConf.load(config_path)
  else:
    config = OmegaConf.load(config_path / 'config.yaml')
  return config
  
def get_last_iter_ckpt_path(run_path):
  model_ckpt_dir = Path(run_path) / 'checkpoints'
  pt_fns = list(model_ckpt_dir.glob('*.pt'))
  if 'last_checkpoint.pt' in [x.name for x in pt_fns]:
    return model_ckpt_dir / 'last_checkpoint.pt'
  sorted_pt_fns = sorted(pt_fns, key=lambda x: int(x.stem.split('_')[0].replace('iter', '')))
  return sorted_pt_fns[-1]
  
def convert_wandb_style_config_to_omega_config(wandb_conf):
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
