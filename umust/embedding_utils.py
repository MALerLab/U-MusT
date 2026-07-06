import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

from pathlib import Path
from typing import Union


class LookUpEmbedding(nn.Module):
  def __init__(self, vq_emb, special_token_embeddings, freeze_vq_emb):
    super().__init__()
    self.special_token_embeddings = nn.Parameter(special_token_embeddings.clone())
    
    if freeze_vq_emb == False:
      self.vq_emb = nn.Parameter(vq_emb.clone())
    else:
      self.vq_emb = vq_emb.clone()
    
  def forward(self, idx):
    self.vq_emb = self.vq_emb.to(idx.device)
    return F.embedding(idx, torch.cat([self.vq_emb, self.special_token_embeddings], dim=0).to(idx.device))

class VQFreezableEmbedding(nn.Module):
  def __init__(self, vq_emb, freeze_vq_emb, dim, num_special_tokens):
    super().__init__()
    
    # # Not normalizing
    # vq_emb = vq_emb
    # # mean = vq_emb.mean(dim=0)
    # # std = vq_emb.std(dim=0)
    # special_token_embeddings = torch.stack([torch.normal(mean, std) for _ in range(self.num_special_tokens)], dim=0)
    
    # Normalizing
    vq_emb = (vq_emb - vq_emb.mean(dim=0)) / vq_emb.std(dim=0) # normalize
    special_token_embeddings = torch.normal(0, 1, size=(num_special_tokens, vq_emb.shape[-1]))
    
    input_embedder = LookUpEmbedding(vq_emb, special_token_embeddings, freeze_vq_emb)
    
    if vq_emb.shape[-1] == dim:
      self.input_embedder = input_embedder
    else:
      self.input_embedder = nn.Sequential(input_embedder, nn.Linear(vq_emb.shape[-1], dim))
    
  def forward(self, idx):
    return self.input_embedder(idx)
  
class MultiEmbedding(nn.Module):
  def __init__(
    self, 
    vocab,
    dim_model,
    weight=None # weight arg is not used, just for compatibility
  ):
    super().__init__()
    '''
    vocab_size: dict of vocab size for each embedding layer
    input_keys: list of input keys
    emb_param: dict of embedding size for each embedding layer
    '''
    self.vocab_size = vocab.vocab_size
    self.feature_list = vocab.feature_list
    self.dim_model = dim_model
    self.layers = []

    self._make_emb_layers()
    self._init_params()
    self._make_emb_boundaries_by_key()
  
  def _init_params(self):
    # apply kaiming init
    for layer in self.layers:
      if isinstance(layer, nn.Embedding):
        nn.init.kaiming_normal_(layer.weight)

  def _make_emb_layers(self):
    vocab_sizes = [self.vocab_size[key] for key in self.feature_list]
    self.embedding_sizes = [self.dim_model for _ in self.feature_list]
    for vocab_size, embedding_size in zip(vocab_sizes, self.embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def _make_emb_boundaries_by_key(self):
    '''
    This function returns dict of boundaries for each embedding layer
    '''
    self.emb_boundary_by_key = {}
    start_idx = 0
    for key, emb_size in zip(self.feature_list, self.embedding_sizes):
      if emb_size != 0:
        self.emb_boundary_by_key[key] = (start_idx, start_idx + emb_size)
        start_idx += emb_size

  def forward(self, x):
    emb = torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)
    return emb

  def __len__(self):
    return len(self.layers)

  def get_emb_by_key(self, key, token):
    '''
    key: key of musical info
    token: B x T (idx of musical info)
    '''
    layer_idx = self.feature_list.index(key)
    return self.layers[layer_idx](token)


class SummationEmbedding(MultiEmbedding):
  def __init__(self, vocab, dim_model, weight=None):
    # weight arg is not used, just for compatibility
    super().__init__(vocab, dim_model)

  def forward(self, seq):
    '''
    seq: B x T x num_features
    '''
    emb_list = [module(seq[..., i]) for i, module in enumerate(self.layers)]
    stacked_emb = torch.stack(emb_list, dim=2) # B x T x num_features x emb_size
    output = torch.sum(stacked_emb, dim=2) # B x T x emb_size
    return output


class RVQMultiEmbedding(nn.Module):
  def __init__(self, vocab, dim_model, weight=None):
    # weight arg is not used, just for compatibility
    
    super().__init__()
    self.vocab_size = vocab.vocab_size
    self.n_codebook = vocab.n_codebook
    self.dim_model = dim_model
    self.features = vocab.feature_list
    self.layers = []
    self._make_emb_layers()

  def _make_emb_layers(self):
    vocab_sizes = [self.vocab_size[key] for key in self.features]
    self.embedding_sizes = [self.dim_model for _ in self.features]
    for vocab_size, embedding_size in zip(vocab_sizes, self.embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    embeddings = torch.zeros(x.shape[0], x.shape[1], self.dim_model).to(x.device)
    emb_list = [module(x[:, (idx+1)%self.n_codebook::self.n_codebook]) for idx, module in enumerate(self.layers)]
    for idx, emb in enumerate(emb_list):
      embeddings[:, (idx+1)%self.n_codebook::self.n_codebook] = emb
    return embeddings
  
  def get_emb_by_key(self, key:str, token:torch.Tensor):
    layer_idx = self.features.index(key)
    return self.layers[layer_idx](token)



class LSEmbedding(nn.Module):
  def __init__(
    self, 
    emb_dim:int, 
    emb_size:int, # n_codebook + n_special_tokens
  ):
    super().__init__()
    
    self.emb_dim = emb_dim
    self.emb_size = emb_size
    
    self.codebook = nn.Embedding(emb_size, emb_dim)
  
  
  def forward(self, emb_id_seq:torch.Tensor):
    """
    emb_id_seq: (N, T)
    """
    z = self.codebook(emb_id_seq) # (N, T, codebook_dim)
    
    return z
  
  
  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False
  
  
  def load_weight_from_emb_tensor(self, emb:torch.Tensor):
    """
    emb: (emb_size, dac_emb_dim)
    """
    # check emb already has special tokens
    if emb.shape[0] == self.emb_size:
      return None
    
    # get number of special tokens
    # difference between self.emb_size and input emb size is the number of special tokens
    n_special_tokens = self.emb_size - emb.shape[0]
    
    # add special token embs to dac_emb
    special_token_embs = self._make_special_token_embs(emb, n_special_tokens)
    emb_w_special_tokens = torch.cat([emb, special_token_embs], dim=0)
    
    self.codebook.weight = nn.Parameter(emb_w_special_tokens)
  
  
  def _make_special_token_embs(self, emb:torch.Tensor, n_special_tokens:int):
    special_token_emb = torch.empty(n_special_tokens, self.emb_dim)
    
    # init special token embeddings with dac embeddings' mean and std
    init_mean = emb.mean() # TODO: pull mean for each dimesion .min(0)
    init_std = emb.std()
    nn.init.normal_(special_token_emb, mean=init_mean, std=init_std)
    
    return special_token_emb



def get_reduced_emb(emb:torch.Tensor, target_dim:int=512):
  """
  emb: (emb_size, n_codebook, emb_dim)
  """
  n_codebook = emb.shape[1]
  
  emb_reduced = []
  for cb_i in range(n_codebook):
    c_emb = emb[:, cb_i, :] # (emb_size, emb_dim)
    
    c_emb_pca = PCA(
      n_components=target_dim, 
      svd_solver='auto', 
      tol=0.0,
      random_state=0,
    ).fit( c_emb.detach().numpy() )
    
    c_emb_reduced = c_emb_pca.transform(c_emb.detach().numpy())
    c_emb_reduced = torch.tensor(c_emb_reduced)
    emb_reduced.append(c_emb_reduced)
  
  emb_reduced = torch.stack(emb_reduced, dim=1) # (emb_size, n_codebook, target_dim)
  
  return emb_reduced



class LSMultiEmbedding(nn.Module):
  def __init__(
    self, 
    vocab,
    dim_model:int,
    weight:Union[torch.tensor,Path,None]=None,
  ):
    """
    vocab: DACVocab
    dim_model: int
    init_weight: (vocab_size, n_codebook, emb_dim)
    """
    super().__init__()
    
    self.emb_dim = dim_model
    self.n_emb = vocab.n_codebook
    self.emb_size = vocab.codebook_size
    self.num_special_tokens = vocab.num_special_tokens
    
    self.quantizers = nn.ModuleList([
      LSEmbedding(
        emb_dim=self.emb_dim,
        emb_size=self.emb_size + self.num_special_tokens,
      )
      for _ in range(self.n_emb)
    ])
    
    # load init weight to each quantizer
    if weight is not None:
      if isinstance(weight, Path):
        weight = torch.load(weight, map_location='cpu')
      
      for i, quantizer in enumerate(self.quantizers):
        quantizer.load_weight_from_emb_tensor(weight[:, i, :])
  
  
  def forward(self, x):
    """
    seq: (N, T, n_codebook)
    """
    x_emb = [ quantizer(x[..., i]) for i, quantizer in enumerate(self.quantizers) ] # [ N, T, emb_dim ] * n_codebook
    x_emb = torch.stack(x_emb, dim=2) # [N, T, n_codebook, emb_dim]
    
    return x_emb
  
  
  def freeze(self):
    for quantizer in self.quantizers:
      quantizer.freeze()
  
  
  def get_config_dict(self):
    return {
      'emb_dim': self.emb_dim,
      'n_emb': self.n_emb,
      'emb_size': self.emb_size,
    }
  
  
  def save(self, path:Union[Path,str]):
    pt = { 
      'configs': self.get_config_dict(),
      'weights': self.state_dict(),
    }
    
    torch.save(pt, path)
  
  
  @classmethod
  def make_from_pt(cls, pt:Union[dict,Path]):
    if isinstance(pt, Path):
      pt = torch.load(pt, map_location='cpu')
    
    kargs = pt['configs']
    state_dict = pt['weights']
    
    embedder = cls(**kargs)
    embedder.load_state_dict(state_dict)
    
    return embedder
  
  
  @classmethod
  def make_from_RVQ(cls, rvq):
    embedder = cls(
      emb_dim=rvq.quantizers[0].out_proj.out_channels,
      n_codebook=len(rvq.quantizers),
      codebook_size=rvq.codebook_size,
      codebook_dim=rvq.quantizers[0].codebook.embedding_dim,
    )
    
    for i, quantizer in enumerate(rvq.quantizers):
      # copy weights
      embedder.quantizers[i].codebook.load_state_dict(quantizer.codebook.state_dict())
      embedder.quantizers[i].out_proj.load_state_dict(quantizer.out_proj.state_dict())
    
    # freeze weights
    embedder.freeze()
    
    return embedder



class LSSummationEmbedding(LSMultiEmbedding):
  def __init__(
    self, 
    vocab,
    dim_model:int,
    weight:Union[torch.tensor,Path,None]=None,
  ):
    super().__init__(vocab, dim_model, weight)
  
  
  def forward(self, x):
    x_emb = super().forward(x) # [N, T, n_codebook, emb_dim]
    x_emb = x_emb.sum(dim=2) # [N, T, emb_dim]
    
    return x_emb
  
class MultimodalRVQEmbedding(nn.Module):
  def __init__(self, num_emb, emb_dim):
    super().__init__()
    self.embeddings = nn.Embedding(num_emb, emb_dim)
    self.proj = None # nn.Sequential(nn.Linear(emb_dim, emb_dim//4), nn.Linear(emb_dim//4, emb_dim))
  
  def forward(self, x):
    assert x.ndim == 3
    x = self.embeddings(x) 
    if self.proj is not None:
      x = self.proj(x)
    x = x.sum(dim=2)
    return x
  
  def load_weight_from_rvq_emb(self, emb, vocab, key, normalize=True):
    hidden_size = self.embeddings.weight.shape[-1]
    token_start_idx = vocab.idx_shifts[key]

    if normalize:
      # Store each codebook's original std
      original_stds = [emb[i].std().item() for i in range(vocab.vocabs[key].n_codebook)]
      print(f"Original standard deviations: {original_stds}")
      
      # Use codebook 0's statistics as the reference
      base_mean = emb[0].mean()
      base_std = emb[0].std()
      
      # Compute each codebook's relative scale
      relative_scales = [std / base_std for std in original_stds]
      print(f"Relative scales to first codebook: {relative_scales}")

    for i in range(vocab.vocabs[key].n_codebook):
      assert emb[i].ndim == 2, f"Embedding dim should be 2, but got {emb[i].ndim}"
      rvq_dim = emb[i].shape[-1]
      if hidden_size % rvq_dim == 0:
        n_repeat = hidden_size // rvq_dim
        if i == 0:
          if n_repeat > 1:
            print(f"Repeating RVQ embedding {n_repeat} times. vq_dim: {rvq_dim} to hidden_size: {hidden_size} dim")
          else:
            print(f"No need to repeat RVQ embedding. vq_dim: {rvq_dim} is already equal to hidden_size: {hidden_size} dim")
        emb[i] = torch.cat([emb[i]] * n_repeat, dim=-1)
      elif rvq_dim % hidden_size == 0:
        n_repeat = rvq_dim // hidden_size
        if i == 0:
          print(f"Truncating RVQ embedding. vq_dim: {rvq_dim} to hidden_size: {hidden_size} dim")
        emb[i] = emb[i][:,:hidden_size]
      else:
        raise ValueError(f"Unsupported hidden size for using RVQ embedding: {hidden_size} // Must be divisible by RVQ dim : {rvq_dim}")
      
      assert vocab.vocabs[key].codebook_size == len(emb[i]), f"RVQ embedding length({len(emb[i])}) should be the same as the codebook size({vocab.vocabs[key].codebook_size})"

      if normalize:
        # Preserve the original relative scale
        print(f"Scaling the normalized {i}th RVQ embedding by relative scale: {relative_scales[i]}")
        emb[i] = ((emb[i] - emb[i].mean()) / emb[i].std()) * relative_scales[i]

      rq_shift = vocab.rq_shifter[key][i]
      with torch.no_grad():
        self.embeddings.weight.data[token_start_idx + rq_shift:token_start_idx + rq_shift + len(emb[i])] = emb[i].clone()

    del emb
    
  
  