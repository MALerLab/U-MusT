import torch
import torch.nn as nn

from .encoders import *
from .decoders import *

class LatentScoreAMT(nn.Module):
  def __init__(self, nn_params, vocab, vq_emb=None):
    super().__init__()
    self.enc_name = nn_params.encoder_params.name
    
    if self.enc_name == "ConvStackGRUEncoder":
      self.encoder = AudioRNNEncoderWrapper(nn_params.encoder_params, nn_params.dropout_shared, nn_params.model_dim)
    elif self.enc_name == "ConvStackTFEncoder" or self.enc_name == "ONFTFEncoder":
      self.encoder = AudioTFEncoderWrapper(nn_params.encoder_params, nn_params.dropout_shared, nn_params.model_dim)
    elif self.enc_name == "YMT3PlusEncoder":
      self.encoder = YMT3PlusEncoderWrapper(nn_params.encoder_params)
    else:
      raise ValueError(f"Unsupported encoder type: {self.enc_name}")
    
    self.decoder = CrossDecoderAutoregressiveWrapper(
      CrossDecoderWrapper(
        vocab,
        nn_params.model_dim,
        nn_params.decoder_params,
        nn_params.dropout_shared,
        vq_emb=vq_emb,
      )
    )

  def forward(self, spec, seq, spec_mask, target=None):
    if self.enc_name == "ConvStackGRUEncoder":
      enc_out = self.encoder(spec)
    elif self.enc_name == "YMT3PlusEncoder":
      enc_out, spec_mask = self.encoder(spec, spec_mask)
    else:
      enc_out = self.encoder(spec, spec_mask)
    return self.decoder(enc_out, seq, spec_mask, target)
  
  @torch.inference_mode()
  def inference(self, input_spec, condition=None, condition_length=200, 
                sampling_method=None, threshold=None, temperature=1, manual_seed=-1):
    if self.enc_name == "ConvStackGRUEncoder":
      enc_out = self.encoder(input_spec)
    elif self.enc_name == "YMT3PlusEncoder":
      enc_out, spec_mask = self.encoder(input_spec, None)
    else:
      enc_out = self.encoder(input_spec, None)
    return self.decoder.inference(
      enc_out, condition=condition, condition_length=condition_length,
      sampling_method=sampling_method, threshold=threshold, temperature=temperature,
      manual_seed=manual_seed
    )

class PianoRollAMT(nn.Module):
  def __init__(self, nn_params, output_features):
    super().__init__()
    self.encoder = AudioTFEncoderWrapper(nn_params.encoder_params, nn_params.dropout_shared, nn_params.model_dim)
    self.decoder = PianoRollDecoder(nn_params.model_dim, output_features)

  def forward(self, mel, mel_mask):
    enc_out = self.encoder(mel, mel_mask)
    return self.decoder(enc_out)

class LMX2VQAMT(nn.Module):
  def __init__(self, nn_params, lmx_vocab, vq_vocab, vq_emb=None):
    super().__init__()
    self.encoder = LMXEncoderWrapper(
      nn_params.encoder_params,
      nn_params.dropout_shared,
      nn_params.model_dim,
      lmx_vocab,
    )
    self.decoder = CrossDecoderAutoregressiveWrapper(
      CrossDecoderWrapper(
        vq_vocab,
        nn_params.model_dim,
        nn_params.decoder_params,
        nn_params.dropout_shared,
        vq_emb=vq_emb,
      )
    )

  def forward(self, lmx, seq, lmx_mask, target=None):
    enc_out = self.encoder(lmx, lmx_mask)
    return self.decoder(enc_out, seq, lmx_mask, target)

  @torch.inference_mode()
  def inference(self, lmx, condition=None, condition_length=50,
                sampling_method=None, threshold=None, temperature=1, manual_seed=-1):
    enc_out = self.encoder(lmx, None)
    return self.decoder.inference(
      enc_out, condition=condition, condition_length=condition_length,
      sampling_method=sampling_method, threshold=threshold, temperature=temperature,
      manual_seed=manual_seed
    )
    
    
class MultimodalTranslator(nn.Module):
  def __init__(self, nn_params, in_vocab:TokenIdxHandler, out_vocab:TokenIdxHandler, vq_emb=None, dac_emb=None):
    super().__init__()
    self.encoder = MultimodalEncoderWrapper(
      nn_params.encoder_params,
      nn_params.dropout_shared,
      nn_params.model_dim,
      in_vocab,
      len(out_vocab.vocab_keys),
      vq_emb=vq_emb,
      dac_emb=dac_emb,
    )
    if nn_params.encoder_params.name == "T5Encoder":
      decoder_wrapper = T5DecoderWrapper(
        out_vocab,
        nn_params.model_dim,
        nn_params.decoder_params,
        nn_params.dropout_shared,
        vq_emb=vq_emb,
        dac_emb=dac_emb,
      )
    else:
      decoder_wrapper = MultimodalDecoderWrapper(
        out_vocab,
        nn_params.model_dim,
        nn_params.decoder_params,
        nn_params.dropout_shared,
        vq_emb=vq_emb,
        dac_emb=dac_emb,
      )
    self.decoder = MultimodalDecoderAutoregressiveWrapper(decoder_wrapper)
    
    self.in_vocab = in_vocab
    self.out_vocab = out_vocab

  def forward(self, in_modal, in_mask, target_in, target_out, modal_idx, in_pos, target_in_pos):
    enc_out = self.encoder(in_modal, in_mask, modal_idx, in_pos)
    return self.decoder(enc_out, target_in, target_out, in_mask, modal_idx, target_in_pos)
  
  @property
  def device(self):
    return next(self.parameters()).device

  @torch.inference_mode()
  def inference(self, in_modal, in_pos, modal_idx, in_mask=None, token_heights=None,
                sampling_method=None, threshold=None, temperature=1, manual_seed=-1, max_length=None, condition=None):
    in_mask = in_mask.to(self.device) if in_mask is not None else None
    modal_idx = modal_idx.to(self.device)
    in_pos = in_pos.to(self.device)
    enc_out = self.encoder(in_modal.to(self.device), in_mask, modal_idx, in_pos)
    return self.decoder.inference(
      enc_out, in_mask=in_mask, modal_idx=modal_idx, token_heights=token_heights,
      sampling_method=sampling_method, threshold=threshold, temperature=temperature,
      manual_seed=manual_seed, max_length=max_length, condition=condition
    )
    
  def compile_model(self):
    if hasattr(self.encoder.encoder, '_orig_mod'):
      return
    self.encoder.encoder = torch.compile(self.encoder.encoder)
    self.encoder.token_emb = torch.compile(self.encoder.token_emb)

    self.decoder.net.input_embedder = torch.compile(self.decoder.net.input_embedder)
    self.decoder.net.decoder = torch.compile(self.decoder.net.decoder)
    self.decoder.net.pos_emb = torch.compile(self.decoder.net.pos_emb)
    self.decoder.net.sub_decoder = torch.compile(self.decoder.net.sub_decoder)

  def uncompile_model(self):
    if not hasattr(self.encoder.encoder, '_orig_mod'):
      return
    self.encoder.encoder = self.encoder.encoder._orig_mod
    self.encoder.token_emb = self.encoder.token_emb._orig_mod

    self.decoder.net.input_embedder = self.decoder.net.input_embedder._orig_mod
    self.decoder.net.decoder = self.decoder.net.decoder._orig_mod
    self.decoder.net.pos_emb = self.decoder.net.pos_emb._orig_mod
    self.decoder.net.sub_decoder = self.decoder.net.sub_decoder._orig_mod
