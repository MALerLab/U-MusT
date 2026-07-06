import torch
import torch.nn as nn

from x_transformers import Decoder
from x_transformers.x_transformers import AbsolutePositionalEmbedding

from .embedding_utils import MultiEmbedding, RVQMultiEmbedding
from .prediction_strategies_utils import *
from .sampling_utils import sample
from .vocab_utils import TokenIdxHandler
from .nano_gpt import Block


class MultimodalSubDecoder(nn.Module):
  def __init__(
      self, 
      vocab:TokenIdxHandler, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      use_nano_gpt=False,
  ):
    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    self.n_features = vocab.max_n_codebook
    self.pos_enc = AbsolutePositionalEmbedding(dim, vocab.max_n_codebook)
    self.token_emb = nn.Embedding(self.vocab_size, dim)
    self.projection = nn.Linear(dim, self.vocab_size)

    nn.init.zeros_(self.pos_enc.emb.weight)
    
    if use_nano_gpt:
      self.transformer_decoder = Block(dim, self.n_features, heads, dropout, bias=False)
    else:
      self.transformer_decoder = Decoder(
        dim=dim,
        depth=sub_decoder_depth,
        heads=heads,
        attn_dropout=dropout,
        ff_dropout=dropout,
        attn_flash=True
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


  def _prepare_token_embedding_for_teacher_forcing(self, hidden_vec_reshape:torch.Tensor, target:torch.Tensor):
    '''
    hidden_vec_reshape: (B*T) x 1 x d_model
    target: B x T x n_codebook
    '''
    used_target = target[:, :, :-1] # B x T x n_codebook-1
    used_target = used_target.flatten(0, 1) # (B*T) x n_codebook-1
    used_target_emb = self.token_emb(used_target) # (B*T) x n_codebook-1 x d_model
    hidden_vec_repeated = hidden_vec_reshape.repeat(1, self.n_features, 1) # (B*T) x n_codebook x d_model
    hidden_vec_repeated[:, 1:] += used_target_emb # (B*T) x n_codebook x d_model
    return hidden_vec_repeated

  def _mask_logits(self, logits, logit_mask, modal_idx:torch.LongTensor, codebook_idx:int):
    assert modal_idx.ndim == 2
    # masks = self.vocab.masks.to(modal_idx.device)[modal_idx[:,1], codebook_idx:codebook_idx+1]
    masks = logit_mask[modal_idx[:,1], codebook_idx:codebook_idx+1]
    return logits.masked_fill(~masks, float('-inf'))

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x n_codebook
    modal_idx = input_dict.get('modal_idx', None)
    # prev_target = input_dict.get('prev_target', None)
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    # input_seq_list = self._prepare_input_seq_list(hidden_vec_reshape, target)

    # ---- Generate(Inference) ---- #
    if target is None:
      input_seq_tensor = hidden_vec_reshape
      logit_list = []
      sampled_token_list = []
      logit_mask = self.vocab.masks.to(modal_idx.device)
      # n_features = self.n_features if not (modal_idx[:,1]<2).all() else 1 # modal_idx 0, 1 is lmx and midi. These have no codebook
      n_features = self.n_features
      for idx in range(n_features):
        tensor_with_pos_enc = input_seq_tensor + self.pos_enc(input_seq_tensor)
        output = self.transformer_decoder(tensor_with_pos_enc)
        logit = self.projection(output[:, -1:, :])
        logit = self._mask_logits(logit, logit_mask, modal_idx, idx)
        logit_list.append(logit)
        # TODO: implement sampling
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_list.append(sampled_token)
        if idx == n_features-1:
          if n_features == 1:
            logit_list.extend([self._mask_logits(torch.zeros_like(logit), logit_mask, modal_idx, codebook_idx=1)] * (self.n_features-1))
            sampled_token_list.extend([torch.zeros_like(sampled_token)] * (self.n_features-1))
          return logit_list, sampled_token_list
        feature_emb = self.token_emb(sampled_token) # d_model
        # feature_emb = self.token_emb(prev_target[:, idx:idx+1].to(sampled_token.device))
        feature_emb += hidden_vec_reshape
        input_seq_tensor = torch.cat([input_seq_tensor, feature_emb], dim=1)
      raise ValueError('Should not reach here')
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(hidden_vec_reshape, target) # (B*T) x (num_features) x d_model
    pos_target_tensor = input_seq_tensor + self.pos_enc(input_seq_tensor) # (B*T) x (num_features) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    logits = self.projection(output)
    logits = logits.reshape((hidden_vec.shape[0], hidden_vec.shape[1], self.n_features, -1))
    return logits

class PredictionStrategy(nn.Module):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__()
    '''
    self.prediction_order: list of token types to be predicted in order,
    token types can be str(in case of sequential prediction) or a list(in case of parallel prediction)
    '''
    self.prediction_order = prediction_order
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    # make layers
    self._make_emb_layer(vocab, dim)
    self._make_projection_layer(vocab, dim)
    self._make_nonlinear_layer()

  @property
  def device(self):
    return next(self.parameters()).device

  def _make_emb_layer(self, vocab, dim):
    self.emb_layer = MultiEmbedding(
      vocab=vocab,
      dim_model=dim
    )

  def _make_projection_layer(self, vocab, dim):
    vocab_sizes = vocab.vocab_size
    self.hidden2logit = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim, size) 
      for key, size in enumerate(vocab_sizes)
    })

  def _make_nonlinear_layer(self):
    pass

  def _seq_sampling(self, prob, target, feature):
    feature_idx = self.config.nn_params.input_keys.index(feature) # TODO: make it faster
    feature_token = target[..., feature_idx] # B x T
    return feature_token
  
  def _parallel_sampling(self, prob_dict, target, feature):
    feature_token_dict = {}
    if target is None:
      for key in feature:
        feature_token = torch.multinomial(prob_dict[key][:, -1, :], num_samples=1)
        feature_token_dict[key] = feature_token
    else: # training
      for key in feature:
        feature_idx = self.config.nn_params.input_keys.index(key)
        feature_token = target[..., feature_idx]
        feature_token_dict[key] = feature_token
    return feature_token_dict

class Parallel_Strategy(PredictionStrategy):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']

    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      for feature in self.prediction_order:
        logit = self.hidden2logit[f"layer_{feature}"](hidden_vec) # B x T x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    for feature in self.prediction_order:
      logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
      logits_dict[feature] = logit
    return logits_dict

class SelfAttention_Strategy(PredictionStrategy):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(1 + len(prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)
    
    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    
    window_size = 1 # number of previous hidden vector of tokens from the main decoder
    causal_mask = generate_causality_mask_on_window(size=window_size + len(prediction_order), window_size=window_size)
    self.register_buffer('causal_mask', causal_mask)

    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = sub_decoder_depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True)
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

  def _apply_pos_enc(self, tgt, apply_type='last'):
    if apply_type == 'all':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      tgt_pos = tgt + self.pos_enc(pos.long())
    elif apply_type == 'last':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      pos_emb = self.pos_enc(pos.long()) # (B*T) x (window_size + BOS + num_features-1) x dim
      # zero out the pos_emb except for the last token
      pos_emb[:, :-1, :] = 0
      tgt_pos = tgt + pos_emb
    return tgt_pos

  def _prepare_input_seq_list(self, hidden_vec_reshape, target=None):
    input_seq_list = []
    input_seq_list.append(hidden_vec_reshape)
    BOS_emb = self.sub_decoder_BOS_emb.unsqueeze(0).repeat(hidden_vec_reshape.shape[0], 1, 1) # (B*T) x 1 x d_model
    if target is None:
      input_seq_list.append(BOS_emb[-1:, :, :])
    else: # training
      input_seq_list.append(BOS_emb)
    return input_seq_list

  def _prepare_token_embedding_for_teacher_forcing(self, input_seq_list, target):
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS + 7) x d_model
    return memory_tensor

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x 8
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq_list = self._prepare_input_seq_list(hidden_vec_reshape, target)
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      input_seq_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS) x d_model
      pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      for idx, feature in enumerate(self.prediction_order):
        output = self.transformer_decoder(pos_target_tensor)
        logit = self.hidden2logit[f"layer_{feature}"](output[:, -1:])
        logits_dict[feature] = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        input_seq_list.append(feature_emb_reshape)
        input_seq_tensor = torch.cat(input_seq_list, dim=1)
        pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='last')
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(input_seq_list, target) # (B*T) x (window_size + BOS + num_features-1) x d_model
    pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS + num_features-1) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict
  
class UniAudioSelfAttention_Strategy(SelfAttention_Strategy):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(1 + len(prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)
    
    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    
    window_size = 1 # number of previous hidden vector of tokens from the main decoder
    causal_mask = generate_causality_mask_on_window(size=window_size + len(prediction_order), window_size=window_size)
    self.register_buffer('causal_mask', causal_mask)

    self.transformer_decoder = Decoder(
      dim=dim,
      depth=sub_decoder_depth,
      heads=heads,
      attn_dropout=dropout,
      ff_dropout=dropout,
      attn_flash=True
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

  def _apply_pos_enc(self, tgt, apply_type='last'):
    if apply_type == 'all':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      tgt_pos = tgt + self.pos_enc(pos.long())
    elif apply_type == 'last':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      pos_emb = self.pos_enc(pos.long()) # (B*T) x (window_size + BOS + num_features-1) x dim
      # zero out the pos_emb except for the last token
      pos_emb[:, :-1, :] = 0
      tgt_pos = tgt + pos_emb
    return tgt_pos

  def _prepare_input_seq_list(self, hidden_vec_reshape, target=None):
    input_seq_list = []
    input_seq_list.append(hidden_vec_reshape)
    BOS_emb = self.sub_decoder_BOS_emb.unsqueeze(0).repeat(hidden_vec_reshape.shape[0], 1, 1) # (B*T) x 1 x d_model
    if target is None:
      input_seq_list.append(BOS_emb[-1:, :, :])
    else: # training
      input_seq_list.append(BOS_emb)
    return input_seq_list

  def _prepare_token_embedding_for_teacher_forcing(self, input_seq_list, target):
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS + 7) x d_model
    return memory_tensor

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x 8
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq_list = self._prepare_input_seq_list(hidden_vec_reshape, target)

    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      input_seq_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS) x d_model
      pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      for idx, feature in enumerate(self.prediction_order):
        output = self.transformer_decoder(pos_target_tensor)
        logit = self.hidden2logit[f"layer_{feature}"](output[:, -1:])
        logits_dict[feature] = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        input_seq_list.append(feature_emb_reshape)
        input_seq_tensor = torch.cat(input_seq_list, dim=1)
        pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='last')
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(input_seq_list, target) # (B*T) x (window_size + BOS + num_features-1) x d_model
    pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS + num_features-1) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict
    
class CrossAttention_Strategy(PredictionStrategy):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.sub_decoder_enricher_use = sub_decoder_enricher_use
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(len(self.prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)

    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    if sub_decoder_enricher_use:
      self.enricher_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    causal_mask = generate_SA_mask(len(prediction_order))
    causl_ca_mask = generate_CA_mask(len(prediction_order), len(prediction_order)).to(self.device)
    self.register_buffer('causal_mask', causal_mask)
    self.register_buffer('causal_ca_mask', causl_ca_mask)

    self.sub_decoder_layers = nn.Sequential(DecoderLayer(dim=dim, num_heads=heads, dropout=dropout))
    if sub_decoder_enricher_use:
      self.feature_enricher_layers = nn.Sequential(FeatureEnricher(dim=dim, num_heads=heads, dropout=dropout))

  def _apply_window_on_hidden_vec(self, hidden_vec):
    BOS_emb = self.enricher_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    # window_size = self.net_param.decoding_attention.decout_window_size
    window_size = 1
    zero_vec = torch.zeros((hidden_vec.shape[0], window_size-1, hidden_vec.shape[2])).to(self.device) # B x (window_size-1) x d_model
    cat_hidden_vec = torch.cat([zero_vec, hidden_vec], dim=1) # B x (window_size-1+T) x d_model
    new_hidden_vec = cat_hidden_vec.unfold(1, window_size, 1).transpose(2, 3) # B x T x window_size x d_model
    new_hidden_vec = new_hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], window_size, -1)) # (B*T) x window_size x d_model
    new_hidden_vec = torch.cat([BOS_emb, new_hidden_vec], dim=1) # (B*T) x (window_size+1) x d_model
    return new_hidden_vec

  def _apply_pos_enc(self, tgt):
    pos = torch.arange(tgt.shape[1]).to(tgt.device) # 8
    pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1) # (B*T) x 8
    tgt_pos = tgt + self.pos_enc(pos.long()) # (B*T) x 8 x d_model
    return tgt_pos

  def _prepare_token_embedding_for_teacher_forcing(self, memory_list, target):
    for _, feature in enumerate(self.prediction_order[:-1]):
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      memory_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + num_features-1) x d_model
    return memory_tensor

  def _prepare_memory_list(self, hidden_vec, target=None):
    memory_list = [] # used for key and value in cross attention
    BOS_emb = self.sub_decoder_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    if target is not None: # training
      memory_list.append(BOS_emb)
    else: # inference
      memory_list.append(BOS_emb[-1:, :, :])
    return memory_list

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target']

    # apply window on hidden_vec for enricher
    if self.sub_decoder_enricher_use:
      window_applied_hidden_vec = self._apply_window_on_hidden_vec(hidden_vec) # (B*T) x window_size x d_model
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x 8 x d_model
    input_seq_pos = self._apply_pos_enc(input_seq)
    # prepare memory
    memory_list = self._prepare_memory_list(hidden_vec=hidden_vec, target=target)
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x 1 x d_model
      for idx, feature in enumerate(self.prediction_order):
        feature_pos = self.feature_order_in_output[feature]
        if self.sub_decoder_enricher_use:
          input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec[-1:]}
          input_dict = self.feature_enricher_layers(input_dict)
          memory_tensor = input_dict['input_seq']
        CA_attn_mask = generate_CA_mask(input_seq_pos.shape[1], memory_tensor.shape[1]).to(self.device)
        input_dict = {'input_seq': input_seq_pos[-1:], 'memory': memory_tensor, 'memory_mask': CA_attn_mask}
        input_dict = self.sub_decoder_layers(input_dict)
        attn_output = input_dict['input_seq']
        logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
        logit = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        memory_list.append(feature_emb_reshape)
        memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + idx+1) x d_model
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    memory_tensor = self._prepare_token_embedding_for_teacher_forcing(memory_list, target) # (B*T) x (BOS + num_features-1) x d_model
    # apply feature enricher to memory
    if self.sub_decoder_enricher_use:
      input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
      input_dict = self.feature_enricher_layers(input_dict)
      memory_tensor = input_dict['input_seq'] # (B*T) x num_features x d_model
    # implement sub decoder cross attention
    input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
    input_dict = self.sub_decoder_layers(input_dict)
    attn_output = input_dict['input_seq'] # (B*T) x num_features x d_model
    # get prob
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class Flatten_Strategy(PredictionStrategy):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def forward(self, input_dict):
    hidden_vec = input_dict['hidden_vec']

    # ---- Training ---- #
    logits_tensor = torch.zeros(hidden_vec.shape[0], hidden_vec.shape[1], self.vocab.codebook_size + self.vocab.num_special_tokens).to(self.device)
    for idx, feature_type in enumerate(self.prediction_order):
      # ::4 means that we only use the first token in each 4 tokens
      # so the chosen tokens will be: 0, 4, 8, 12, ...
      # 1::4 means that we only use the second token in each 4 tokens
      # so the chosen tokens will be: 1, 5, 9, 13, ...
      separated_hidden_vec = hidden_vec[:, idx::self.vocab.n_codebook, :]
      logit = self.hidden2logit[f"layer_{feature_type}"](separated_hidden_vec)
      logits_tensor[:, idx::self.vocab.n_codebook, :] = logit
      # prob_dict[feature_type] = prob
    return logits_tensor
  
  def run_one_step(self, input_dict, sampling_method=None, threshold=None, temperature=None, feature_type=None):
    # ---- Generate(Inference) ---- #
    hidden_vec = input_dict['hidden_vec']
    logit = self.hidden2logit[f"layer_{feature_type}"](hidden_vec[:, -1:])
    sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    return logit, sampled_token



class LMX_Strategy(nn.Module):
  def __init__(
    self, 
    prediction_order, 
    vocab, 
    sub_decoder_depth, 
    dim, 
    heads, 
    dropout,
    sub_decoder_enricher_use
  ):
    super().__init__()
    
    self.vocab = vocab
    self.dim = dim
    self.dropout = dropout
    self.hidden2logit = nn.Linear(dim, vocab.vocab_size)
  
  
  def forward(self, input_dict):
    """
    **for training**
    input_dict: { 'hidden_vec', 'input_seq', 'target' }
    """
    hidden_vec = input_dict['hidden_vec']
    logit = self.hidden2logit(hidden_vec)
    
    return logit
  
  
  def run_one_step(self, input_dict, sampling_method=None, threshold=None, temperature=None, feature_type=None):
    """
    **for inference**
    input_dict: { 'hidden_vec', 'input_seq', 'target' }
    """
    
    hidden_vec = input_dict['hidden_vec']
    logit = self.hidden2logit(hidden_vec)
    sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    
    return logit, sampled_token