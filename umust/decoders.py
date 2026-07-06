import torch
import torch.nn as nn
import x_transformers
from tqdm import tqdm
from x_transformers.x_transformers import LayerIntermediates

from .embedding_utils import VQFreezableEmbedding, MultimodalRVQEmbedding
from .vocab_utils import VQVocab, LMXVocab, RVQVocab, TokenIdxHandler
from .pos_utils import TwoDimPosEmbedding, PosEmbedding, MultimodalFlattenedPosEmbedding
from .sampling_utils import sample
from .prediction_strategies import MultimodalSubDecoder
from . import prediction_strategies
from . import embedding_utils

class XtransformerCrossDecoder(nn.Module):
  def __init__(
      self, 
      dim,
      depth,
      heads,
      dropout,
  ):
    super().__init__()
    self._make_decoder_layer(dim, depth, heads, dropout)
    
  def _make_decoder_layer(self, dim, depth, heads, dropout):
    self.transformer_decoder = x_transformers.Decoder(
                                    dim = dim,
                                    depth = depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True,
                                    cross_attend = True,
                                    )
    # add final dropout
    print('Applying Xavier Uniform Init to x-transformer following torch.Transformer')
    self._apply_xavier_init()
    print('Adding dropout after feedforward layer in x-transformer')
    self._add_dropout_after_ff(dropout)
    print('Adding dropout after attention layer in x-transformer')
    self._add_dropout_after_attn(dropout)

  def _add_dropout_after_attn(self, dropout):
    for layer in self.transformer_decoder.layers:
      if 'Attention' in str(type(layer[1])): 
        if isinstance(layer[1].to_out, nn.Sequential): # if GLU
          layer[1].to_out.append(nn.Dropout(dropout))
        elif isinstance(layer[1].to_out, nn.Linear): # if simple linear
          layer[1].to_out = nn.Sequential(layer[1].to_out, nn.Dropout(dropout))
        else:
          raise ValueError('to_out should be either nn.Sequential or nn.Linear')

  def _add_dropout_after_ff(self, dropout):
    for layer in self.transformer_decoder.layers:
      if 'FeedForward' in str(type(layer[1])):
        layer[1].ff.append(nn.Dropout(dropout))

  def _apply_xavier_init(self):
    for name, param in self.transformer_decoder.named_parameters():
      if 'to_q' in name or 'to_k' in name or 'to_v' in name:
          torch.nn.init.xavier_uniform_(param, gain=0.5**0.5)

  def forward(self, enc_out, seq, enc_out_mask=None, cache=None):
    if cache is not None: # implementing run_one_step in inference
      if cache.hiddens is None: cache = None
      hidden_vec, intermediates = self.transformer_decoder(seq, context=enc_out, context_mask=enc_out_mask, cache=cache, return_hiddens=True)
      return hidden_vec, intermediates
    else:
      return self.transformer_decoder(seq, context=enc_out, context_mask=enc_out_mask)


class DimDropout(nn.Module):
  def __init__(self, p:float, dim_list):
    """
      p:  probability of an element to be zero-ed.
      dim_list: list of tuple of slice object eg. [(slice(None), slice(1, 2), slice(None))]
      eg. ex_tensor[(slice(None), slice(1, 2), slice(None))] == ex_tensor[:, 1:2, :]
    """
    super().__init__()
    self.p = torch.tensor(p)
    self.dim_list = dim_list
  
  
  def forward(self, x):
    clone = x.clone()
    
    if self.training:
      for dim in self.dim_list:
        mask = torch.bernoulli(torch.ones(clone[dim].shape[:2]) * (1-self.p))
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, clone.shape[-1])
        mask = mask.to(clone.device)
        
        clone[dim] *= mask
    
    return clone

class PianoRollDecoder(nn.Module):
  def __init__(self, hidden_size, output_features):
    super().__init__()
    self.output_features = output_features
    self.decoder = nn.Linear(hidden_size, output_features * 2)
  
  def forward(self, enc_out):
    out =  self.decoder(enc_out)
    out = out.permute(0, 2, 1)
    return out.reshape([out.shape[0], 2, self.output_features, -1])
  


class MultimodalDecoderWrapper(nn.Module):
  def __init__(self, vocab:TokenIdxHandler, dim, decoder_params, dropout_shared, vq_emb=None, dac_emb=None):
    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    self.emb_dropout = nn.Dropout(decoder_params.decoder.get('dropout', dropout_shared))
    self.max_tok_len = self.vocab.max_tok_len
    self.max_token_height = self.vocab.max_token_height

    # # Main Decoder Teacher-forcing Dropout
    # self.tf_dropout = DimDropout(decoder_params.tf_dropout, [(slice(None), slice(1, None), slice(None))]) 

    self.main_norm = nn.LayerNorm(dim)
    
    self.input_embedder = MultimodalRVQEmbedding(self.vocab_size, dim)

    if dac_emb is not None and 'dac' in vocab.vocab_keys:
      print(f"Loading DAC embedding for decoder...")
      if self.input_embedder.embeddings.weight.shape[-1] % dac_emb[0].shape[-1] != 0:
        print(f"Resizing token_emb.embeddings.weight to match dac_emb[0].shape[-1]")
        print(f"token_emb.embeddings.weight.shape[-1]: {self.input_embedder.embeddings.weight.shape[-1]}")
        self.input_embedder.embeddings = nn.Embedding(self.vocab_size, dac_emb[0].shape[-1]) # rvq is 256, and dac is 1024

      self.input_embedder.load_weight_from_rvq_emb(dac_emb, vocab, 'dac', normalize=True)
      print(f"Decoder DAC embedding loaded successfully")
      del dac_emb
    if vq_emb is not None and 'pt' in vocab.vocab_keys:
      print(f"Loading VQ embedding for decoder...")
      self.input_embedder.load_weight_from_rvq_emb(vq_emb, vocab, 'pt', normalize=True)
      print(f"Decoder VQ embedding loaded successfully")
      del vq_emb
      
    if self.input_embedder.embeddings.weight.shape[-1] != dim:
      self.input_embedder.proj = nn.Sequential(nn.Linear(self.input_embedder.embeddings.weight.shape[-1], dim//4), nn.Linear(dim//4, dim))
      print(f"MultimodalDecoderWrapper embedding projection layer added")
      
    self.pos_emb = MultimodalFlattenedPosEmbedding(dim, self.vocab.pos_idx_size, self.vocab.pad_idx, self.vocab.vocab_keys)
    self.sub_decoder = MultimodalSubDecoder(
      vocab=vocab,
      sub_decoder_depth=decoder_params.sub_decoder.num_layer,
      dim=dim,
      heads=decoder_params.sub_decoder.num_head,
      dropout=dropout_shared,
    )
    
    self.decoder = XtransformerCrossDecoder(
      dim=dim,
      depth=decoder_params.decoder.num_layer,
      heads=decoder_params.decoder.num_head,
      dropout=decoder_params.decoder.get('dropout', dropout_shared),
    )
    

  def forward(self, enc_out:torch.Tensor, input_seq:torch.Tensor, target_out:torch.Tensor, enc_out_mask:torch.BoolTensor, modal_idx:torch.Tensor, input_pos:torch.Tensor):
    # modal_idx: B x 2, where the first column is the index of the input modality and the second column is the index of the output modality
    # input_pos: torch.Tensor of shape (B, T, 2) where each element is the position of the token. dummy 2 for pt(x,y)
    embedding = self.input_embedder(input_seq)
    embedding = embedding + self.pos_emb(embedding.shape, input_pos, modal_idx[:,1])
    embedding = self.emb_dropout(embedding)
    
    hidden_vec = self.decoder(enc_out, embedding, enc_out_mask=enc_out_mask)
    hidden_vec = self.main_norm(hidden_vec)
    logits = self.sub_decoder({'hidden_vec': hidden_vec, 'target': target_out})
    return logits
  
  @property
  def device(self):
    return next(self.parameters()).device

class T5DecoderWrapper(MultimodalDecoderWrapper):
  def __init__(self, vocab:TokenIdxHandler, dim, decoder_params, dropout_shared, vq_emb=None, dac_emb=None):
    emb_preloaded = (vq_emb is not None and 'pt' in vocab.vocab_keys) or (dac_emb is not None and 'dac' in vocab.vocab_keys)
    super().__init__(vocab, dim, decoder_params, dropout_shared, vq_emb, dac_emb)
    from transformers import T5ForConditionalGeneration
    t5_model = T5ForConditionalGeneration.from_pretrained(decoder_params.model_name)
    self.decoder = t5_model.decoder
    self.decoder.embed_tokens = None
    self.sub_decoder = MultimodalSubDecoder(
      vocab=vocab,
      sub_decoder_depth=decoder_params.sub_decoder.num_layer,
      dim=dim,
      heads=decoder_params.sub_decoder.num_head,
      dropout=dropout_shared,
      use_nano_gpt=True,
    )

    del self.main_norm

  def forward(self, enc_out:torch.Tensor, input_seq:torch.Tensor, enc_out_mask:torch.BoolTensor, modal_idx:torch.Tensor, input_pos:torch.Tensor):
    embedding = self.input_embedder(input_seq)
    embedding = embedding + self.pos_emb(embedding.shape, input_pos, modal_idx[:,1])
    embedding = self.emb_dropout(embedding)
    
    hidden_vec = self.decoder(inputs_embeds=embedding, encoder_hidden_states=enc_out['last_hidden_state'], encoder_attention_mask=enc_out_mask)
    logits = self.sub_decoder({'hidden_vec': hidden_vec['last_hidden_state'], 'target': input_seq})
    return logits

class MultimodalDecoderAutoregressiveWrapper(nn.Module):
  def __init__(self, net:MultimodalDecoderWrapper):
    super().__init__()
    self.net:MultimodalDecoderWrapper = net

  def forward(self, enc_out:torch.Tensor, input_seq:torch.LongTensor, target_out:torch.LongTensor, enc_out_mask:torch.BoolTensor, modal_idx:torch.Tensor, input_pos:torch.Tensor):
    return self.net(enc_out, input_seq, target_out, enc_out_mask, modal_idx, input_pos)
  
  def _prepare_inference(self, modal_idx:torch.LongTensor, manual_seed, condition=None):
    if manual_seed > 0:
      torch.manual_seed(manual_seed)
    total_out = self.net.vocab.prepare_start_token(modal_idx)
    if condition is not None:
      total_out = torch.cat([total_out, condition], dim=1)
    return total_out

  def _run_one_step(self, enc_out, input_seq, enc_mask, modal_idx, input_pos, cache=None, sampling_method=None, threshold=None, temperature=1):
    embedding = self.net.input_embedder(input_seq)
    embedding = embedding + self.net.pos_emb(embedding.shape, input_pos, modal_idx[:,1])
    hidden_vec, intermidiates = self.net.decoder(enc_out, embedding, enc_mask, cache=cache) # B x T x d_model
    hidden_vec = self.net.main_norm(hidden_vec)
    hidden_vec = hidden_vec[:, -1:] # B x 1 x d_model
    input_dict = {'hidden_vec': hidden_vec, 'target': None, 'modal_idx': modal_idx}
    logit_list, sampled_token_list = self.net.sub_decoder(input_dict, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    logits = torch.stack(logit_list, dim=-1)
    sampled_token = torch.stack(sampled_token_list, dim=-1)
    return logits, sampled_token, intermidiates
  
  def _update_total_out(self, total_out, sampled_token):
    # if self.net.vocab.encoding_scheme == 'flatten':
    # TODO:batch-inf: check this modification works
    if sampled_token.ndim == 1:
      sampled_token = sampled_token.unsqueeze(0)
    total_out = torch.cat([total_out, sampled_token], dim=1) # B(1) x T 
    return total_out, sampled_token
  
  @torch.inference_mode()
  def inference(self, enc_out, in_mask, modal_idx, token_heights, sampling_method=None, threshold=None, temperature=1, manual_seed=-1, max_length=None, verbose=False, condition=None):
    total_out = self._prepare_inference(modal_idx, manual_seed, condition=condition).to(enc_out.device)
    
    cache = LayerIntermediates()
      
    # TODO:batch-inf: replace max_seq_len to decoder maximum input length
    is_ended = torch.zeros(total_out.shape[0], dtype=torch.bool).to(self.net.device)  
    max_seq_len = max_length if max_length is not None else self.net.max_tok_len
    eos_of_batch = self.net.vocab.shifted_eos_tensors.to(modal_idx.device)[modal_idx[:,1]]
    
    y_pos = self.net.vocab.make_pos_emb_from_tensor(total_out, modal_idx, token_heights).to(self.net.device)
    pbar = tqdm(total=max_seq_len) if verbose else None
    # prev_target = total_out[:,-1]
    while total_out.shape[1] < max_seq_len:
      logits, sampled_token, cache = self._run_one_step(enc_out, total_out, in_mask, modal_idx, y_pos, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      total_out, sampled_token = self._update_total_out(total_out, sampled_token) # TODO:batch-inf: check this modification works
      is_ended += (sampled_token == eos_of_batch).all(dim=-1).squeeze(1) # .any() is faster than .all()
      
      if is_ended.all():
        break
      y_pos = self.net.vocab.make_pos_emb_from_tensor(total_out, modal_idx, token_heights).to(self.net.device)
      if pbar is not None:
        pbar.update(1)
      # prev_target = total_out[:,-1]
    if pbar is not None:
      pbar.close()
    is_eos = (total_out[:,1:] == eos_of_batch).all(dim=-1)
    mask = is_eos.cumsum(dim=1) >= 1
    mask = mask[:,:-1]

    pad_idx = self.net.vocab.pad_idx
    # Create a mask for tokens after the first <eos> in each sequence
    # Replace tokens after <eos> with <pad>
    total_out[:,2:][mask] = pad_idx

    return total_out


class CrossDecoderWrapper(nn.Module):
  def __init__(self, vocab, dim, decoder_params, dropout_shared, vq_emb=None):
    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    self.pad_token = vocab.pad_idx if hasattr(vocab, 'pad_idx') else None
    self.start_token = vocab.sos_idx if hasattr(vocab, 'sos_idx') else None
    self.end_token = vocab.eos_idx if hasattr(vocab, 'eos_idx') else None
    self.max_tok_width = decoder_params.max_tok_width
        
    self.emb_dropout = nn.Dropout(decoder_params.decoder.get('dropout', dropout_shared))
    
    # Main Decoder Teacher-forcing Dropout
    self.tf_dropout = DimDropout(decoder_params.tf_dropout, [(slice(None), slice(1, None), slice(None))]) 

    self.main_norm = nn.LayerNorm(dim)
    
    if not isinstance(self.vocab, RVQVocab):
      self.projection = nn.Linear(dim, self.vocab_size)
    
    if isinstance(self.vocab, VQVocab):
      self.dec_pos_enc = TwoDimPosEmbedding(dim, self.vocab.token_height, self.max_tok_width)
      if isinstance(self.vocab, RVQVocab):
        self.input_embedder = getattr(embedding_utils, decoder_params.input_embedder.name)(
          vocab=vocab,
          dim_model=dim,
        )
        self.sub_decoder = getattr(prediction_strategies, decoder_params.sub_decoder.name)(
          prediction_order=self.vocab.feature_list,
          vocab=vocab,
          dim=dim,
          sub_decoder_depth=decoder_params.sub_decoder.num_layer,
          heads=decoder_params.sub_decoder.num_head,
          dropout=dropout_shared,
          sub_decoder_enricher_use=decoder_params.sub_decoder.feature_enricher_use,
        )
      else:
        if vq_emb is not None:
          self.input_embedder = VQFreezableEmbedding(vq_emb, decoder_params.freeze_vq_emb, dim, self.vocab.num_special_tokens)
          del vq_emb
        else:
          self.input_embedder = nn.Embedding(self.vocab_size, dim)
    elif isinstance(self.vocab, LMXVocab):
      self.dec_pos_enc = PosEmbedding(dim, self.max_tok_width)
      self.input_embedder = nn.Embedding(self.vocab_size, dim)
    else:
      raise ValueError(f"Unsupported vocab type: {type(self.vocab)}")
    
    self.decoder = XtransformerCrossDecoder(
      dim=dim,
      depth=decoder_params.decoder.num_layer,
      heads=decoder_params.decoder.num_head,
      dropout=decoder_params.decoder.get('dropout', dropout_shared),
    )
    

  def forward(self, enc_out:torch.Tensor, input_seq:torch.Tensor, enc_out_mask, target=None):
    embedding = self.input_embedder(input_seq)
    embedding = self.tf_dropout(embedding)
    
    if isinstance(self.vocab, VQVocab):
      embedding = torch.cat([embedding[:,:1], self.dec_pos_enc(embedding[:,1:].reshape([embedding.shape[0], -1, self.vocab.token_height, embedding.shape[-1]]).transpose(1,2)).transpose(1,2).flatten(1,2)], dim=-2)
    elif isinstance(self.vocab, LMXVocab):
      embedding = self.dec_pos_enc(embedding)
    else:
      raise ValueError(f"Unsupported vocab type: {type(self.vocab)}")
    
    embedding = self.emb_dropout(embedding)
    hidden_vec = self.decoder(enc_out, embedding, enc_out_mask=enc_out_mask)
    hidden_vec = self.main_norm(hidden_vec)
    if not isinstance(self.vocab, RVQVocab):
      logits = self.projection(hidden_vec)
    else:
      input_dict = {'hidden_vec': hidden_vec, 'target': target}
      logits = self.sub_decoder(input_dict)
    return logits
  
  @property
  def device(self):
    return next(self.parameters()).device

class CrossDecoderAutoregressiveWrapper(nn.Module):
  def __init__(self, net:CrossDecoderWrapper):
    super().__init__()
    self.net = net

  def forward(self, enc_out:torch.Tensor, input_seq:torch.Tensor, enc_out_mask, target=None):
    return self.net(enc_out, input_seq, enc_out_mask, target)
  
  def _prepare_inference(self, start_token, manual_seed, condition=None, condition_length=200):
    if manual_seed > 0:
      torch.manual_seed(manual_seed)
      
    total_out = []
    if condition is None:
      if isinstance(self.net.vocab, RVQVocab):
        total_out.append([start_token] * self.net.vocab.n_codebook)
      else:
        total_out.append(start_token)
      total_out = torch.LongTensor(total_out).unsqueeze(0)
    else:
      selected_tokens = condition[:,:condition_length].tolist()
      total_out.extend(selected_tokens)
      total_out = torch.LongTensor(total_out)
    return total_out

  def _run_one_step(self, enc_out, input_seq, cache=None, sampling_method=None, threshold=None, temperature=1):
    embedding = self.net.input_embedder(input_seq)
    if isinstance(self.net.vocab, VQVocab):
      pad = torch.zeros([embedding.shape[0], (self.net.vocab.token_height - embedding.shape[1] + 1) % self.net.vocab.token_height, embedding.shape[-1]], device=embedding.device)
      embedding = torch.cat([embedding[:,:1], self.net.dec_pos_enc(torch.cat([embedding[:,1:],pad], dim=-2).reshape([embedding.shape[0], -1, self.net.vocab.token_height, embedding.shape[-1]]).transpose(1,2)).transpose(1,2).flatten(1,2)[:,:embedding.shape[1]-1]], dim=-2)
    elif isinstance(self.net.vocab, LMXVocab):
      embedding = self.net.dec_pos_enc(embedding)
    else:
      raise ValueError(f"Unsupported vocab type: {type(self.net.vocab)}")
    
    hidden_vec, intermidiates = self.net.decoder(enc_out, embedding, cache=cache) # B x T x d_model
    hidden_vec = self.net.main_norm(hidden_vec)
    hidden_vec = hidden_vec[:, -1:] # B x 1 x d_model
    if isinstance(self.net.vocab, RVQVocab):
      input_dict = {'hidden_vec': hidden_vec, 'target': None}
      logits_dict, sampled_token_dict = self.net.sub_decoder(input_dict, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      logits = torch.stack([logits_dict[key] for key in self.net.vocab.feature_list], dim=-2)
      sampled_token = torch.stack([sampled_token_dict[key] for key in self.net.vocab.feature_list], dim=-1).unsqueeze(0)
    else:
      logits = self.net.projection(hidden_vec)
      sampled_token = sample(logits, sampling_method, threshold, temperature)
    return logits, sampled_token, intermidiates
  
  def _update_total_out(self, total_out, sampled_token):
    # if self.net.vocab.encoding_scheme == 'flatten':
    # TODO:batch-inf: check this modification works
    if sampled_token.ndim == 1:
      sampled_token = sampled_token.unsqueeze(0)
    total_out = torch.cat([total_out, sampled_token], dim=1) # B(1) x T 
    return total_out, sampled_token
  
  @torch.inference_mode()
  def inference(self, enc_out, condition=None, condition_length=None, sampling_method=None, threshold=None, temperature=1, manual_seed=-1):
    total_out = self._prepare_inference(self.net.start_token, manual_seed, condition=condition, condition_length=condition_length).to(enc_out.device)
    
    if condition is not None:
      _, _, cache = self._run_one_step(enc_out, total_out, cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    else:
      cache = LayerIntermediates()
      
    # TODO:batch-inf: replace max_seq_len to decoder maximum input length
    is_ended = torch.zeros(total_out.shape[0], dtype=torch.bool).to(self.net.device)  
    
    if isinstance(self.net.vocab, VQVocab) or isinstance(self.net.vocab, RVQVocab):
      max_seq_len = self.net.max_tok_width * self.net.vocab.token_height
    elif isinstance(self.net.vocab, LMXVocab):
      max_seq_len = self.net.max_tok_width
    else:
      raise ValueError(f"Unsupported vocab type: {type(self.net.vocab)}")
    
    while total_out.shape[1] < max_seq_len:
      _, sampled_token, cache = self._run_one_step(enc_out, total_out, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      total_out, sampled_token = self._update_total_out(total_out, sampled_token) # TODO:batch-inf: check this modification works
      # print("sampled_token", sampled_token)
      # print("total_out", total_out)
      # print("sampled_token", sampled_token)
      # print("self.net.end_token", self.net.end_token)
      is_ended += (sampled_token == self.net.end_token).any(dim=-1).squeeze(1) # .any() is faster than .all()
      
      if is_ended.all():
        break
      
      # if isinstance(self.net.vocab, RVQVocab):
      #   is_ended += (sampled_token.tolist() == [self.net.end_token] * self.net.vocab.n_codebook).squeeze(1)
      #   print("is_ended", is_ended)
      #   if is_ended.all():
      #     break
        
      # if (isinstance(self.net.vocab, VQVocab) and not isinstance(self.net.vocab, RVQVocab)) or isinstance(self.net.vocab, LMXVocab):
      #   if self.net.end_token in sampled_token.tolist():
      #     break
      
      # if self.net.end_token in sampled_token.tolist():
      #   break
      
    return total_out
