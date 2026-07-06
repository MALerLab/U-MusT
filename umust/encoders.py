import torch.nn as nn
import torch.nn.functional as F
import torch

import x_transformers

from .pos_utils import SinusPosEncoding, PosEmbedding, MultimodalFlattenedPosEmbedding
from .constants import *
from .vocab_utils import TokenIdxHandler
from .embedding_utils import MultimodalRVQEmbedding

import sys

from .yourmt3plus.model.conv_block import PreEncoderBlockRes3B
from .yourmt3plus.model.perceiver_mod import PerceiverTFEncoder, PerceiverTFConfig
from .yourmt3plus.model.spectrogram import get_spectrogram_layer_from_audio_cfg
#ConvStackLSTMEncoder(N_MELS, 768, 2, 0.5)

class ConvStack(nn.Module):
  def __init__(self, input_features, output_features):
    super().__init__()

    # input is batch_size * 1 channel * frames * input_features
    self.cnn = nn.Sequential(
      # layer 0
      nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
      nn.BatchNorm2d(output_features // 16),
      nn.ReLU(),
      # layer 1
      nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
      nn.BatchNorm2d(output_features // 16),
      nn.ReLU(),
      # layer 2
      nn.MaxPool2d((2, 1)),
      nn.Dropout(0.25),
      nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
      nn.BatchNorm2d(output_features // 8),
      nn.ReLU(),
      # layer 3
      nn.MaxPool2d((2, 1)),
      nn.Dropout(0.25),
    )
    self.fc = nn.Sequential(
      nn.Linear((output_features // 8) * (input_features // 4), output_features),
      nn.Dropout(0.5)
    )

  def forward(self, mel):
    x = mel.unsqueeze(1)
    # x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
    # x = x.transpose(-2, -1)
    x = self.cnn(x) # N C F T
    x = x.permute(0, 3, 1, 2).flatten(-2) # N T CF
    x = self.fc(x)
    return x

class ConvStackGRUEncoder(nn.Module):
  def __init__(self, input_features, hidden_size, num_layers, dropout):
    super().__init__()
    
    self.conv = ConvStack(input_features, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size//2, num_layers, dropout=dropout, bidirectional=True, batch_first=True) # bidirectional GRU

  def forward(self, mel):
    x = self.conv(mel)
    x = self.gru(x)
    return x[0]

class ConvStackTFEncoder(nn.Module):
  def __init__(self, input_features, hidden_size, heads, num_layers, dropout, max_seq):
    super().__init__()
    
    self.conv = ConvStack(input_features, hidden_size)
    self.transformer = x_transformers.Encoder(dim=hidden_size, depth=num_layers, heads=heads, attn_dropout=dropout, ff_dropout=dropout, attn_flash=True)
    self.enc_pos_enc = SinusPosEncoding(hidden_size, max_seq)
    
  def forward(self, mel, mel_mask):
    x = self.conv(mel)
    x = self.enc_pos_enc(x)
    x = self.transformer(x, mask=mel_mask)
    return x

class ONFTFEncoder(nn.Module): # OnsetsAndFramesTF
    def __init__(self, input_features, hidden_size, heads, num_layers, dropout, max_seq=3000):
        super().__init__()
        sequence_model = lambda hidden_size, num_layers, heads, dropout: x_transformers.Encoder(dim=hidden_size, depth=num_layers, heads=heads, attn_dropout=dropout, ff_dropout=dropout, attn_flash=True)

        self.conv = ConvStack(input_features, hidden_size)
        self.enc_pos_enc = SinusPosEncoding(hidden_size, max_seq)
        self.transformer = sequence_model(hidden_size, num_layers, heads, dropout)

    def forward(self, mel, mel_mask):
        x = self.conv(mel)
        x = self.enc_pos_enc(x)
        x = self.transformer(x, mask=mel_mask)
        return x
      
class LMXEncoder(nn.Module):
  def __init__(self, hidden_size, heads, num_layers, dropout, max_seq):
    super().__init__()
    self.transformer = x_transformers.Encoder(dim=hidden_size, depth=num_layers, heads=heads, attn_dropout=dropout, ff_dropout=dropout, attn_flash=True)
    self.enc_pos_enc = PosEmbedding(hidden_size, max_seq)

  def forward(self, lmx, lmx_mask):
    x = self.enc_pos_enc(lmx)
    x = self.transformer(x, mask=lmx_mask)
    return x

class LMXEncoderWrapper(nn.Module):
  def __init__(self, encoder_params, dropout_shared, hidden_size, vocab):
    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    
    self.lmx_token_emb = nn.Embedding(self.vocab_size, hidden_size)
    self.encoder = getattr(sys.modules[__name__], encoder_params.name)(
      hidden_size,
      encoder_params.num_head,
      encoder_params.num_layer,
      encoder_params.get('dropout', dropout_shared),
      encoder_params.max_lmx_length
    )

  def forward(self, lmx, lmx_mask):
    lmx = self.lmx_token_emb(lmx)
    return self.encoder(lmx, lmx_mask)

class AudioRNNEncoderWrapper(nn.Module):
  def __init__(self, encoder_params, dropout_shared, hidden_size):
    super().__init__()
    self.encoder = getattr(sys.modules[__name__], encoder_params.name)(
      N_MELS,
      hidden_size,
      encoder_params.num_layer,
      encoder_params.get('dropout', dropout_shared)
    )
    self.enc_pos_enc = SinusPosEncoding(hidden_size, encoder_params.max_mel_time_bin)

  def forward(self, mel_spec):
    enc_out = self.encoder(mel_spec)
    enc_out = enc_out + self.enc_pos_enc(enc_out)
    return enc_out
  
class AudioTFEncoderWrapper(nn.Module):
  def __init__(self, encoder_params, dropout_shared, hidden_size):
    super().__init__()
    self.encoder = getattr(sys.modules[__name__], encoder_params.name)(
      N_MELS,
      hidden_size,
      encoder_params.num_head,
      encoder_params.num_layer,
      encoder_params.get('dropout', dropout_shared),
      encoder_params.max_mel_time_bin
    )
    
    if hasattr(encoder_params, 'from_pretrained') and encoder_params.from_pretrained:
      print(f"Loading pretrained weights from {encoder_params.from_pretrained}")
      self.load_state_dict(torch.load(encoder_params.from_pretrained), strict=False)
    
  def forward(self, mel_spec, mel_mask):
    enc_out = self.encoder(mel_spec, mel_mask)
    return enc_out

class YMT3PlusEncoderWrapper(nn.Module):
  def __init__(self, encoder_params):
    super().__init__()

    # Initialize encoder components
    self.pre_encoder = nn.Sequential(PreEncoderBlockRes3B(1, 
                                          encoder_params["conv_out_channels"],
                                          kernel_size=(3,3),
                                          avp_kernerl_size=(2,2))
    )
    
    perceivertf_config = PerceiverTFConfig()
    perceivertf_config.update(encoder_params["encoder"])
    self.encoder = PerceiverTFEncoder(perceivertf_config)

    self.summary_vector = nn.Parameter(torch.randn(1, 1, encoder_params["encoder"]["d_model"]))
    self.summary_attn = nn.MultiheadAttention(embed_dim=encoder_params["encoder"]["d_model"], num_heads=1, batch_first=True)

    self.out_pos_enc = SinusPosEncoding(encoder_params["encoder"]["d_model"], encoder_params["encoder"]["num_max_positions_out"])

    if encoder_params.from_pretrained:
      print(f"Loading pretrained weights from {encoder_params.from_pretrained}")
      self.load_state_dict(torch.load(encoder_params.from_pretrained), strict=False)
  
  def downsample_mask(self, mask, original_len=110, target_len=13):
    # mask shape: [B,N,110]
    B, N, T = mask.shape
    
    # Convert bool to float
    mask = mask.float()
    
    # Add channel dimension for pooling
    mask = mask.unsqueeze(1)  # [B,N,1,110]
    
    kernel_size = (original_len + target_len - 1) // target_len
    stride = original_len // target_len
    
    pooled_mask = F.max_pool2d(
      mask, 
      kernel_size=(1, kernel_size),
      stride=(1, stride),
      padding=(0, 0)
    )
    
    # Remove extra dimension and convert back to bool
    pooled_mask = pooled_mask.squeeze(1)  # [B,N,13]
    pooled_mask = pooled_mask > 0.5  # Convert back to bool

    return pooled_mask

  def to_attention_mask(self, mask):
    BN, T = mask.shape
    K = self.encoder.latent_array.latents.size(0)
    H = self.encoder.blocks[0].temporal_transformer[0].attention.self.num_heads
    mask = mask.repeat(K, 1)
    mask = mask.unsqueeze(1).unsqueeze(2) # [BNK, 1, 1, T]
    mask = mask & mask.transpose(-1, -2)  # [BNK, 1, T, T]
    mask = (~mask).float() * -10000.0
    return mask
  
  def forward(self, spec, spec_mask=None):
    if spec.dim() == 3:
      spec = spec.unsqueeze(0)
      spec_mask = spec_mask.unsqueeze(0)
    B, N, T, F = spec.shape
    spec = spec.reshape(B*N, T, F)
  
    # Pre-encoder
    x = self.pre_encoder(spec)  # project to d_model

    # Downsample mask
    downsampled_mask = None
    if spec_mask is not None:
      downsampled_mask = self.downsample_mask(spec_mask, T, x.shape[1])
      spec_mask = downsampled_mask.reshape(B*N, -1)
      spec_mask = self.to_attention_mask(spec_mask)
      downsampled_mask = downsampled_mask.flatten(-2,-1)

    # Encoder
    x = self.encoder(inputs_embeds=x, temporal_attention_mask=spec_mask)["last_hidden_state"]

    BN, T, K, D = x.shape
    x = x.reshape(B*N*T, K, D)

    x = self.summary_attn(query=self.summary_vector.expand(B*T*N, -1, -1), key=x, value=x)[0]
    x = x.reshape(B, N*T, D)

    x = x + self.out_pos_enc(x)
    return x, downsampled_mask

class MultimodalEncoder(nn.Module):
  def __init__(self, hidden_size, heads, num_layers, dropout, vocab:TokenIdxHandler, num_out_modalities:int):
    super().__init__()
    self.vocab = vocab
    self.max_tok_len = self.vocab.max_tok_len

    self.transformer = x_transformers.Encoder(dim=hidden_size, depth=num_layers, heads=heads, attn_dropout=dropout, ff_dropout=dropout, attn_flash=True)
    self.pos_emb = MultimodalFlattenedPosEmbedding(hidden_size, self.vocab.pos_idx_size, self.vocab.pad_idx, self.vocab.vocab_keys)
    self.out_modal_emb = nn.Embedding(num_out_modalities, hidden_size)

  def forward(self, x, mask, modal_idx, x_pos):
    # modal_idx: torch.Tensor of shape (batch_size) where each element is the index of the modality
    # x_pos: torch.Tensor of shape (batch_size, seq_len, 2) where each element is the position of the token. dummy 2 for pt(x,y)
    x = x + self.pos_emb(x.shape, x_pos, modal_idx[:,0])
    x = x + self.out_modal_emb(modal_idx[:,1]).unsqueeze(1) # add embedding of output modality
    x = self.transformer(x, mask=mask)
    return x
  
class T5Encoder(MultimodalEncoder):
  def __init__(self, hidden_size, heads, num_layers, dropout, vocab:TokenIdxHandler, t5_name="t5-small"):
    super().__init__(hidden_size, heads, num_layers, dropout, vocab)
    from transformers import T5EncoderModel
    t5encoder = T5EncoderModel.from_pretrained(t5_name)
    self.transformer = t5encoder.encoder
    self.transformer.embed_tokens = None
    
  def forward(self, x, mask, modal_idx, x_pos):
    x = x + self.pos_emb(x.shape, x_pos, modal_idx[:,0])
    x = x + self.out_modal_emb(modal_idx[:,1]).unsqueeze(1) # add embedding of output modality
    x = self.transformer(inputs_embeds=x, attention_mask=mask)
    return x

class MultimodalEncoderWrapper(nn.Module):
  def __init__(self, encoder_params, dropout_shared, hidden_size, vocab:TokenIdxHandler, num_out_modalities:int, vq_emb=None, dac_emb=None, compile=False):
    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    
    self.token_emb = MultimodalRVQEmbedding(self.vocab_size, hidden_size)
    emb_preloaded = False
    if dac_emb is not None and 'dac' in vocab.vocab_keys:
      print(f"Loading DAC embedding for encoder...")
      if self.token_emb.embeddings.weight.shape[-1] % dac_emb[0].shape[-1] != 0:
        print(f"Resizing token_emb.embeddings.weight to match dac_emb[0].shape[-1]")
        print(f"token_emb.embeddings.weight.shape[-1]: {self.token_emb.embeddings.weight.shape[-1]}")
        self.token_emb.embeddings = nn.Embedding(self.vocab_size, dac_emb[0].shape[-1]) # rvq is 256, and dac is 1024
      self.token_emb.load_weight_from_rvq_emb(dac_emb, vocab, 'dac', normalize=True)
      del dac_emb
      emb_preloaded = True
      print(f"Encoder DAC embedding loaded successfully\n")

    if vq_emb is not None and 'pt' in vocab.vocab_keys:
      print(f"Loading VQ embedding for encoder...")
      self.token_emb.load_weight_from_rvq_emb(vq_emb, vocab, 'pt', normalize=True)
      
      print(f"Encoder VQ embedding loaded successfully\n")
      del vq_emb
      emb_preloaded = True
    
    if self.token_emb.embeddings.weight.shape[-1] != hidden_size:
      self.token_emb.proj = nn.Sequential(nn.Linear(self.token_emb.embeddings.weight.shape[-1], hidden_size//4), nn.Linear(hidden_size//4, hidden_size))
      print(f"MultimodalEncoderWrapper embedding projection layer added")

    if encoder_params.name == "T5Encoder" and emb_preloaded:
      self.token_emb.proj = nn.Sequential(nn.Linear(hidden_size, hidden_size//4), nn.Linear(hidden_size//4, hidden_size))
      self.token_emb.embeddings.requires_grad_(False)
      print(f"T5Encoder embedding projection layer added")
    if encoder_params.name == "T5Encoder":
      self.encoder = T5Encoder(
        hidden_size,
        encoder_params.num_head,
        encoder_params.num_layer,
        encoder_params.get('dropout', dropout_shared),
        vocab,
        t5_name=encoder_params.model_name
      )
    else:
      self.encoder = MultimodalEncoder(
        hidden_size,
        encoder_params.num_head,
        encoder_params.num_layer,
        encoder_params.get('dropout', dropout_shared),
        vocab,
        num_out_modalities
        )
    if compile: 
      self.encoder = torch.compile(self.encoder)
      self.token_emb = torch.compile(self.token_emb)


  def forward(self, x, mask, modal_idx, x_pos):
    x = self.token_emb(x)
    return self.encoder(x, mask, modal_idx, x_pos)
