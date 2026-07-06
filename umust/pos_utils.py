import torch
import torch.nn as nn
from x_transformers.x_transformers import AbsolutePositionalEmbedding

class TwoDimPosEmbedding(nn.Module):
  def __init__(self, dim, height_length, width_length=200):
    super().__init__()
    self.emb_y = AbsolutePositionalEmbedding(dim, height_length)
    self.emb_x = AbsolutePositionalEmbedding(dim, width_length)
  
  def forward(self, t):
    pos_y = self.emb_y(t).reshape(1, t.shape[1], 1, t.shape[-1])
    pos_x = self.emb_x(t.transpose(1,2)).reshape(1, 1, t.shape[2], t.shape[-1]) 
    
    t = t + pos_y + pos_x
    
    return t
  
class SinusPosEncoding(nn.Module):
  def __init__(self, emb_size, max_t):
    super().__init__()
    self.emb_size =emb_size
    self.max_t = max_t
    self.register_buffer('encoding', self._prepare_emb())

  def _prepare_emb(self):
    dim_axis = 10000**(torch.arange(self.emb_size//2) * 2 / self.emb_size) # 10000 ** (normalized values between 0~1 num_emb_dim)
    timesteps = torch.arange(self.max_t)
    pos_enc_in = timesteps.unsqueeze(1) / dim_axis.unsqueeze(0)
    pos_enc_sin = torch.sin(pos_enc_in) # x values for sin are between 0 ~ 1 so the values could never be the same
    pos_enc_cos = torch.cos(pos_enc_in)

    pos_enc = torch.stack([pos_enc_sin, pos_enc_cos], dim=-1).reshape([self.max_t, self.emb_size])
    return pos_enc

  def forward(self, x):
    x_t = x.shape[1]
    return x + self.encoding[:x_t, ...]

class PosEmbedding(nn.Module):
  def __init__(self, dim, width_length):
    super().__init__()
    self.emb = AbsolutePositionalEmbedding(dim, width_length)
  
  def forward(self, t):
    return t + self.emb(t)
  
  
class MultimodalPosEmbedding(nn.Module):
  def __init__(self, hidden_size, max_seq, num_modalities=3):
    super().__init__()
    self.layers = nn.ModuleList([AbsolutePositionalEmbedding(hidden_size, max_seq) for _ in range(num_modalities)])
    self.num_modalities = num_modalities
  def forward(self, x, modal_idx):
    pos_emb = torch.zeros_like(x)
    for i in range(self.num_modalities):
      sample_of_idx = (modal_idx == i)
      pos_emb[sample_of_idx] = self.layers[i](x[sample_of_idx])
    return x + pos_emb

class MultimodalFlattenedPosEmbedding(nn.Module):
  def __init__(self, hidden_size, pos_idx_size, pad_idx, vocab_keys):
    super().__init__()
    self.vocab_keys = vocab_keys
    self.pos_emb = nn.Embedding(pos_idx_size, hidden_size, padding_idx=pad_idx)

  def forward(self, x_shape, x_pos, modal_idx):
    pos_emb = torch.zeros(x_shape).to(x_pos.device)
    # for i in range(len(self.vocab_keys)):
    #   sample_of_idx = (modal_idx == i)
    #   pos_emb[sample_of_idx] += self.pos_emb(x_pos[sample_of_idx][...,0].long())
    #   if self.vocab_keys[i] == 'pt':
    #     pos_emb[sample_of_idx] += self.pos_emb(x_pos[sample_of_idx][...,1].long())
    pos_emb = self.pos_emb(x_pos.long()).sum(dim=-2)
    return pos_emb