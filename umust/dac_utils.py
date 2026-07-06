import torch
import torch.nn.functional as F

from typing import Union, List
from pathlib import Path

from dac import DAC, DACFile
from dac.nn.quantize import ResidualVectorQuantize
from dac.model.dac import init_weights
from audiotools import AudioSignal

import tqdm


class LSResidualVectorQuantize(ResidualVectorQuantize):
  def __init__(
    self,
    input_dim: int = 512,
    n_codebooks: int = 9,
    codebook_size: int = 1024,
    codebook_dim: Union[int, list] = 8,
    quantizer_dropout: float = 0.0,
  ):
    super().__init__()


  def from_codes(self, codes: torch.Tensor, n_quantizers: int = None):
    """Given the quantized codes, reconstruct the continuous representation
    Parameters
    ----------
    codes : Tensor[B x N x T]
        Quantized discrete representation of input
    Returns
    -------
    Tensor[B x D x T]
        Quantized continuous representation of input
    """
    n_codebooks = codes.shape[1]
    
    if n_quantizers is not None and n_quantizers < n_codebooks:
      n_codebooks = n_quantizers
    
    z_q = 0.0
    z_p = []
    
    for i in range(n_codebooks):
      z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
      z_p.append(z_p_i)

      z_q_i = self.quantizers[i].out_proj(z_p_i)
      z_q = z_q + z_q_i
    
    return z_q, torch.cat(z_p, dim=1), codes
  
  
  @classmethod
  def make_from_RVQ(cls, rvq):
    lsrvq = cls(
      input_dim=rvq.quantizers[0].in_proj.in_channels,
      n_codebooks=rvq.n_codebooks,
      codebook_size=rvq.codebook_size,
      codebook_dim=rvq.codebook_dim,
      quantizer_dropout=rvq.quantizer_dropout,
    )

    lsrvq.quantizers = rvq.quantizers
    
    return lsrvq



class LSDAC(DAC):
  def __init__(
      self,
      encoder_dim: int = 64,
      encoder_rates: List[int] = [2, 4, 8, 8],
      latent_dim: int = None,
      decoder_dim: int = 1536,
      decoder_rates: List[int] = [8, 8, 4, 2],
      n_codebooks: int = 9,
      codebook_size: int = 1024,
      codebook_dim: Union[int, list] = 8,
      quantizer_dropout: bool = False,
      sample_rate: int = 44100,
  ):
    super().__init__(
      encoder_dim, encoder_rates, 
      latent_dim, 
      decoder_dim, decoder_rates, 
      n_codebooks, codebook_size, codebook_dim, 
      quantizer_dropout, 
      sample_rate
    )
    
    self.quantizer = LSResidualVectorQuantize.make_from_RVQ(self.quantizer)
    self.apply(init_weights)
  
  
  
  @torch.no_grad()
  def decompress(
    self,
    obj: Union[str, Path, DACFile],
    verbose: bool = False,
    n_quantizers: int = None,
  ) -> AudioSignal:
    """
    Reconstruct audio from a given .dac file
    
    Parameters
    ----------
    obj : Union[str, Path, DACFile]
      .dac file location or corresponding DACFile object.
    verbose : bool, optional
      Prints progress if True, by default False

    Returns
    -------
    AudioSignal
      Object with the reconstructed audio
    """
    self.eval()
    if isinstance(obj, (str, Path)):
      obj = DACFile.load(obj)
    
    original_padding = self.padding
    self.padding = obj.padding
    
    range_fn = range if not verbose else tqdm.trange
    codes = obj.codes
    original_device = codes.device
    chunk_length = obj.chunk_length
    recons = []
    
    for i in range_fn(0, codes.shape[-1], chunk_length):
      c = codes[..., i : i + chunk_length].to(self.device)
      z = self.quantizer.from_codes(c, n_quantizers)[0]
      r = self.decode(z)
      recons.append(r.to(original_device))
    
    recons = torch.cat(recons, dim=-1)
    recons = AudioSignal(recons, self.sample_rate)
    
    resample_fn = recons.resample
    loudness_fn = recons.loudness
    
    # If audio is > 10 minutes long, use the ffmpeg versions
    if recons.signal_duration >= 10 * 60 * 60:
      resample_fn = recons.ffmpeg_resample
      loudness_fn = recons.ffmpeg_loudness
    
    recons.normalize(obj.input_db)
    resample_fn(obj.sample_rate)
    recons = recons[..., : obj.original_length]
    loudness_fn()
    recons.audio_data = recons.audio_data.reshape(
        -1, obj.channels, obj.original_length
    )
    
    self.padding = original_padding
    return recons
  
  
  @torch.no_grad()
  def decompress_tensor(
    self,
    codes: torch.Tensor,
    verbose: bool = False,
    n_quantizers: int = None,
    chunk_length:Union[int,None] = None, # DAC default chunk length
  ) -> AudioSignal:
    """
    Reconstruct audio from a given .dac file
    
    Parameters
    ----------
    codes : Tensor
        DAC code tensor
    verbose : bool, optional
        Prints progress if True, by default False
    
    Returns
    -------
    AudioSignal
        Object with the reconstructed audio
    """
    self.eval()
    
    original_padding = self.padding
    self.padding = False
    
    range_fn = range if not verbose else tqdm.trange
    original_device = codes.device
    signal = []
    
    if chunk_length is None:
      chunk_length = codes.shape[-1]
    
    for i in range_fn(0, codes.shape[-1], chunk_length):
      c = codes[..., i : i + chunk_length].to(self.device)
      z = self.quantizer.from_codes(c, n_quantizers)[0]
      r = self.decode(z)
      signal.append(r.to(original_device))
    
    signal = torch.cat(signal, dim=-1)
    signal = AudioSignal(signal, self.sample_rate)
    
    resample_fn = signal.resample
    loudness_fn = signal.loudness
    
    # If audio is > 10 minutes long, use the ffmpeg versions
    if signal.signal_duration >= 10 * 60 * 60:
      resample_fn = signal.ffmpeg_resample
      loudness_fn = signal.ffmpeg_loudness
      
    signal.normalize()
    resample_fn(self.sample_rate)
    loudness_fn()
    
    self.padding = original_padding
    return signal
  
  
  @classmethod
  def make_from_dac_model(cls, dac_model):
    model = cls(
      encoder_dim=dac_model.encoder_dim,
      encoder_rates=dac_model.encoder_rates,
      latent_dim=dac_model.latent_dim,
      decoder_dim=dac_model.decoder_dim,
      decoder_rates=dac_model.decoder_rates,
      n_codebooks=dac_model.n_codebooks,
      codebook_size=dac_model.codebook_size,
      codebook_dim=dac_model.codebook_dim,
      quantizer_dropout=dac_model.quantizer.quantizer_dropout,
      sample_rate=dac_model.sample_rate,
    )
    
    model.load_state_dict(dac_model.state_dict())
    
    return model