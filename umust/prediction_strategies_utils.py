from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

# from x-transformers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class MLP(nn.Module):
  def __init__(self, in_size, out_size, hidden_size, dropout):
    super().__init__()
    self.out_size = out_size
    self.layer = nn.Sequential(
      nn.Linear(in_size, hidden_size),
      nn.Dropout(dropout),
      nn.ReLU(),
      nn.Linear(hidden_size, out_size)
    )

  def forward(self, x):
    return self.layer(x)

class extendedMLP(nn.Module):
  def __init__(self, in_size, out_size, num_layers, hidden_size, dropout):
    super().__init__()
    self.input_size = in_size

    self.layers = nn.ModuleList()
    if num_layers == 1:
      # Only one layer
      self.layers.append(nn.Linear(in_size, out_size))
      return
    elif num_layers > 1:
      # First layer
      self.layers.append(nn.Linear(in_size, hidden_size))
      self.layers.append(nn.Dropout(dropout))
      self.layers.append(nn.ReLU())
      # Intermediate layers
      if num_layers > 2:
        for _ in range(num_layers - 2):  # -2 because we're manually adding the first and last layers
          self.layers.append(nn.Linear(hidden_size, hidden_size))
          self.layers.append(nn.Dropout(dropout))
          self.layers.append(nn.ReLU())
      # Last layer
      self.layers.append(nn.Linear(hidden_size, out_size))
    else:
      raise ValueError("num_layers should be a positive integer")
  
  def forward(self, x):
     for layer in self.layers:
         x = layer(x)
     return x

class multiMLP(nn.Module):
  def __init__(self, in_size, out_size, hidden_size, dropout, pred_order):
    super().__init__()
    self.out_size = out_size
    self.layer = nn.ModuleList([MLP(in_size, out_size, hidden_size, dropout) for _ in pred_order])
  
  def forward(self, x, choice):
    '''
    x: B x T x d_model
    choice: token type from self.pred_order (str or list of str)
    '''
    if isinstance(choice, str):
      idx = self.pred_order.index(choice)
      return self.layer[idx](x)
    elif len(choice) > 1 and not isinstance(choice, str):
      raise ValueError("multiMLP doesn't support parallel prediction")

class ResidualLayerNormModule(nn.Module):
  def __init__(self, submodule: nn.Module):
    super().__init__()
    self.submodule = submodule
    if submodule.__class__.__name__ == 'MultiheadAttention':
      self.layer_norm = nn.LayerNorm(self.submodule.embed_dim)
    else:
      self.layer_norm = nn.LayerNorm(self.submodule.input_size)

  def forward_attention(self, q, k, v, attn_mask, type):
    attn_output, _ = self.submodule(q, k, v, attn_mask=attn_mask, need_weights=False, average_attn_weights=False)
    return self.layer_norm(attn_output + q)

  def forward_mlp(self, x):
    return self.layer_norm(self.submodule(x) + x)

class MultiProj_hidden2logit(nn.Module):
  def __init__(self, dim, vocab_sizes):
    super().__init__()
    self.layers = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim, size) for key, size in vocab_sizes.items()
      })
  
  def forward(self, hidden_vec, feature):
    logit = self.layers[f"layer_{feature}"](hidden_vec)
    return logit

class MultiProj_catvec2hidden(nn.Module):
  def __init__(self, config, par_pred_keys, seq_pred_keys):
    super().__init__()
    '''
    This class is used in SQstyleEachEmbStrategy
    par_pred_keys: list of independent features(These tokens are predicted in parallel)
    seq_pred_keys: list of sequential features(These tokens are predicted sequentially)
    '''
    net_param = config.nn_params
    self.d_model = net_param.model.d_model
    independent_emb_size = 0
    for key in par_pred_keys:
      independent_emb_size += net_param.emb[key]
    self.layers = nn.ModuleDict({
      'layer_independent': nn.Linear(self.d_model + independent_emb_size, self.d_model),
      **{f"layer_{key}": nn.Linear(self.d_model + net_param.emb[key], self.d_model) for key in seq_pred_keys}
      })
    self.par_pred_keys = par_pred_keys
    self.seq_pred_keys = seq_pred_keys
    self.dropout = nn.Dropout(0.1)
    self.relu = nn.ReLU()
  
  def forward(self, x, choice):
    '''
    x: B x T x (d_model + emb_size)
    choice: key type (str or list of str)
    '''
    if isinstance(choice, str): # single key
      assert choice in self.seq_pred_keys
      output = self.layers[f"layer_{choice}"](x)
      return self.relu(self.dropout(output))
    elif len(choice) > 1 and not isinstance(choice, str): # multiple keys, parallel
      assert choice == self.par_pred_keys # the order of choice should be the same as the order of self.par_pred_keys
      output = self.layers['layer_independent'](x)
      return self.relu(self.dropout(output))

def mask_tensor(tensor, mask_rate=0.15):
  # Get the size of the tensor
  batch_size, seq_len, dim = tensor.size()
  # Calculate the total number of elements and the number to mask
  total_elements = batch_size * seq_len
  num_to_mask = int(total_elements * mask_rate)
  # Create a 1D binary mask where 1 indicates that element will be masked.
  # Start by creating a tensor of zeros with length equal to the total number of elements.
  mask = torch.zeros(total_elements).to(tensor.device)
  # Set `num_to_mask` random indices to 1 (masking)
  indices_to_mask = torch.randperm(total_elements)[:num_to_mask]
  mask[indices_to_mask] = 1
  # Reshape the mask to match the original tensor's shape
  mask = mask.reshape(batch_size, seq_len)
  mask = mask.unsqueeze(2) # B x T x 1
  masked_tensor = tensor * (mask == 0).float() # B x T x d_model
  return masked_tensor

def generate_causality_mask_on_window(size, window_size):
  mask = torch.zeros((size, size))
  for i in range(size):
    mask[i, i+window_size:] = 1
  return mask.bool()

# generate boolean mask, if the value is 1 or true, it means the value is masked
# considers BOS token and mask margin
def generate_CA_mask(tgt_len, memory_len, mask_margin=0):
  mask = torch.triu(torch.ones((tgt_len, memory_len)), diagonal=mask_margin+1)
  return mask.bool()

# generate boolean mask, if the value is 1 or true, it means the value is masked
def generate_SA_mask(tgt_len):
  mask = torch.triu(torch.ones((tgt_len, tgt_len)), diagonal=1)
  return mask.bool()

class DecoderLayer(nn.Module):
  def __init__(self, dim, num_heads, dropout):
    super().__init__()
    self.cross_attn_block = ResidualLayerNormModule(nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True))
    self.residual_FF = ResidualLayerNormModule(extendedMLP(in_size=dim, out_size=dim, num_layers=2, hidden_size=2048, dropout=dropout))
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, input_dict):
    '''
    input_dict = {'input_seq': input_seq, 'memory': memory, 'memory_mask': CA_attn_mask}
    '''
    # cross attention
    attn_output = self.cross_attn_block.forward_attention(input_dict['input_seq'], input_dict['memory'], input_dict['memory'], input_dict['memory_mask'], type='cross')
    attn_output = self.residual_FF.forward_mlp(attn_output) 
    attn_output = self.dropout(attn_output)
    output_dict = {'input_seq': attn_output, 'memory': input_dict['memory'], 'memory_mask': input_dict['memory_mask']}
    return output_dict

class FeatureEnricher(nn.Module):
  def __init__(self, dim, num_heads, dropout):
    super().__init__()
    self.cross_attn_block = ResidualLayerNormModule(nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True))
    self.residual_FF = ResidualLayerNormModule(extendedMLP(in_size=dim, out_size=dim, num_layers=2, hidden_size=2048, dropout=dropout))
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, input_dict):
    '''
    input_dict = {'input_seq': input_seq, 'memory': memory}
    '''
    # cross attention
    attn_output = self.cross_attn_block.forward_attention(input_dict['input_seq'], input_dict['memory'], input_dict['memory'], None, type='feature_enrichment')
    attn_output = self.residual_FF.forward_mlp(attn_output)
    attn_output = self.dropout(attn_output)
    output_dict = {'input_seq': attn_output, 'memory': input_dict['memory']}
    return output_dict