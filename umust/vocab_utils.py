from typing import Union, List
from pathlib import Path
import torch

class VQVocab():
  def __init__(
    self, 
    codebook_size:int,
    num_special_tokens:int,
    token_height:int=15,
  ):
    self.num_special_tokens = num_special_tokens
    self.codebook_size = codebook_size
    self.token_height = token_height
    self._prepare_in_vocab()
    self._get_sos_eos_token()

  def _prepare_in_vocab(self):
    codebook_vocab = {}
    for code_idx in range(self.codebook_size):
      codebook_vocab[str(code_idx)]=code_idx
    if self.num_special_tokens == 1:
      codebook_vocab[str(self.codebook_size)]=self.codebook_size # PAD, SOS, EOS
    elif self.num_special_tokens == 2:
      codebook_vocab[str(self.codebook_size)]=self.codebook_size # PAD
      codebook_vocab[str(self.codebook_size+1)]=self.codebook_size+1 # SOS, EOS
    elif self.num_special_tokens == 3:
      codebook_vocab[str(self.codebook_size)]=self.codebook_size # PAD
      codebook_vocab[str(self.codebook_size+1)]=self.codebook_size+1 # SOS
      codebook_vocab[str(self.codebook_size+2)]=self.codebook_size+2 # EOS
    else:
      raise ValueError(f'Invalid number of special tokens: {self.num_special_tokens}')
    self.codebook_vocab = codebook_vocab
    
  def _get_sos_eos_token(self):
    if self.num_special_tokens == 1:
      self.pad_token = self.pad_idx = self.codebook_size
      self.sos_token = self.sos_idx =  self.codebook_size
      self.eos_token = self.eos_idx = self.codebook_size
    elif self.num_special_tokens == 2:
      self.pad_token = self.pad_idx = self.codebook_size
      self.sos_token = self.sos_idx = self.codebook_size+1
      self.eos_token = self.eos_idx = self.codebook_size+1
    elif self.num_special_tokens == 3:
      self.pad_token = self.pad_idx = self.codebook_size
      self.sos_token = self.sos_idx = self.codebook_size+1
      self.eos_token = self.eos_idx = self.codebook_size+2
    else:
      raise ValueError(f'Invalid number of special tokens: {self.num_special_tokens}')

  @property
  def vocab_size(self):
    return len(self.codebook_vocab)
  
  def __len__(self):
    return self.vocab_size

class RVQVocab(VQVocab):
  def __init__(
    self, 
    codebook_size:int,
    num_special_tokens:int,
    token_height:int=15,
    n_codebook:int=4,
  ):
    super().__init__(
      codebook_size=codebook_size,
      num_special_tokens=num_special_tokens,
      token_height=token_height,
    )
    self.n_codebook = n_codebook
    self._get_features()
    
  def _get_features(self):
    self.feature_list = list( range(self.n_codebook) )
  
  @property
  def vocab_size(self):
    return [len(self.codebook_vocab)] * self.n_codebook
  
  def __len__(self):
    return sum(self.vocab_size)


class LMXVocab():
  def __init__(
    self, 
    entire_strs:str=None, 
    vocab_txt_fn:Union[Path,str]=None, 
    num_special_tokens:int=2,
  ) -> None:
    """
    if entire_strs is not None, it will be used to generate vocab.
    if vocab_txt_fn is not None, it will be used to load vocab.
    num_special_tokens:
      if 1: pad == sos == eos
      if 2: pad != sos == eos
      if 3: pad != sos != eos (default)
    """
    # for compatibility with DACVocab
    self.n_codebook = 1
    self.encoding_scheme = 'flatten'
    self.feature_list = [ i for i in range(self.n_codebook) ]
    self.num_special_tokens = num_special_tokens
    self.num_special_tokens = len(self._get_special_tokens())
    
    assert entire_strs is not None or vocab_txt_fn is not None, 'Either entire_strs or vocab_txt_fn must be provided.'
    
    
    self.entire_strs = entire_strs
    self.vocab_txt_fn = vocab_txt_fn
    
    if self.vocab_txt_fn:
      vocabs = self._load_vocab(vocab_txt_fn)
      self.entire_strs = ' '.join(vocabs)
    
    else:
      vocabs = self.get_vocab()
    
    self.vocab = vocabs
    self.tok2idx = { tok: idx for idx, tok in enumerate(vocabs) }
    
    self.pad_idx, self.sos_idx, self.eos_idx = [ self.tok2idx[t] for t in [self.pad_token, self.sos_token, self.eos_token] ]
    
    # self.vocab = [ vocabs for _ in self.feature_list ]
    # self.tok2idx = [ {tok: idx for idx, tok in enumerate(vocabs)} for _ in self.feature_list ]
    # self.vocab_size = [ len(vocabs) for _ in self.feature_list ]
  
  def _get_special_tokens(self) -> List[str]:
    match self.num_special_tokens:
      case 1:
        special_tokens = ['<pad>']
        self.pad_token = self.sos_token = self.eos_token = '<pad>'
      case 2:
        special_tokens = ['<pad>', '<start>']
        self.pad_token = '<pad>'
        self.sos_token = self.eos_token = '<start>'
      case 3:
        special_tokens = ['<pad>', '<start>', '<end>']
        self.pad_token = '<pad>'
        self.sos_token = '<start>'
        self.eos_token = '<end>'
    
    return special_tokens
  
  def _load_vocab(self, vocab_txt_fn:Union[Path,str]) -> List[str]:    
    with open(vocab_txt_fn, 'r') as f:
      vocab = [ l for l in f.read().split('\n') if l != '' ]
    
    assert len(vocab) == len(set(vocab)), 'There are duplicated tokens in vocab file.'
    
    num_special_tokens_from_txt = len( set(vocab[:3]).intersection({'<pad>', '<start>', '<end>'}) )
    
    special_tokens = self._get_special_tokens()
    if num_special_tokens_from_txt < 1:
      print('WARNING: There is no special token in vocab file, could be a major issue.')
      vocab = special_tokens + vocab
    
    elif self.num_special_tokens != num_special_tokens_from_txt:
      raise(ValueError, 'Number of special tokens in vocab file is not matched with propvided num_special_tokens.')
    
    return vocab
  
  
  def get_vocab(self) -> List[str]:
    entire_tokens = (
      self.entire_strs
        .replace('\n', ' ')
        .split(' ')
    )
    
    vocab_set = set(entire_tokens)
    vocab_ls = sorted(list(vocab_set))
    
    special_tokens = self._get_special_tokens()
    
    return special_tokens + vocab_ls
  
  
  # encode input string to list of token indices
  def __call__(self, lmx_str:str) -> List[int]:
    # split lmx string by space
    words = (
      lmx_str
        .replace('\n', '')
        .split(' ')
    )
    words = [word for word in words if word != '']
    
    # add sos and eos tokens
    words = [self.sos_token] + words + [self.eos_token]
    
    # encode words to token indices
    words_encoded = [ self.tok2idx[word] for word in words ]
    
    return words_encoded
  
  
  def __add__(self, other:'LMXVocab') -> 'LMXVocab':
    return LMXVocab(
      entire_strs=' '.join([self.entire_strs, other.entire_strs]),
    )
  
  @property
  def vocab_size(self):
    return len(self.vocab)
  
  
  def _get_special_indices(self) -> list[int]:
    return [ self.pad_idx, self.sos_idx, self.eos_idx ]
  
  
  def decode(self, indices:Union[torch.Tensor, List[int]]) -> str:
    if isinstance(indices, torch.Tensor):
      if indices.ndim == 2: # [1, seq_len]
        indices = indices.squeeze(0) # [seq_len]
      
      indices = indices.tolist()
    
    # slice indices before first eos token
    if self.eos_token in indices:
      indices = indices[:indices.index(self.eos_token)]
    
    special_indices = self._get_special_indices()
    special_indices = set(special_indices)
    
    indices_decoded = [
      self.vocab[idx] for idx in indices 
      if idx not in special_indices # pad, sos, eos
    ]
    
    lmx_decoded = ' '.join(indices_decoded)
    
    return lmx_decoded

  def __len__(self) -> int:
    return len(self.vocab)
  
  
  
class TokenIdxHandler:
  def __init__(self, vocabs: dict, max_seq_len:dict, max_pt_x_len:int=150, out_pt_height_token=False):
    self.vocab_keys = list(vocabs.keys())
    print(f"Preparing idx shifter for {self.vocab_keys}")
    self.max_seq_len = max_seq_len # ["lmx": 800, "pt": 200(width), "dac": 2000]}
    self.max_pt_x_len = max_pt_x_len
    self.vocabs = vocabs
    self.pad_idx = 0 # 0 is the modality-universal pad token
    self.out_pt_height_token = out_pt_height_token
    if 'pt' in self.vocab_keys:
      self.max_token_height = vocabs['pt'].token_height
      self.img_crop_cat_sep_idx = self.vocab_size - 1 # last token is the image crop separator token: used when concatenating image crops
      if self.out_pt_height_token:
        self.pt_height_tokens = [self.vocab_size - 2 - i for i in reversed(range(self.max_token_height))]
    else:
      self.max_token_height = 0
    self.n_codebook = max([vocabs[k].n_codebook for k in self.vocab_keys])
    self.idx_shifts, self.idx_range = self._prepare_idx_shift_by_vocab()
    self.rq_shifter = {k: self._prepare_rq_shifter(vocabs[k]) for k in self.vocab_keys 
                       if isinstance(vocabs[k], RVQVocab) }
    self.pos_shifts, self.pos_range, self.pos_idx_size = self._prepare_pos_shift_by_max_seq_len()
    self.sos_tensors, self.eos_tensors, self.sos_shifted_tensors, self.eos_shifted_tensors = self._prepare_sos_eos_tensors()
    self.masks = self._prepare_logit_mask()
    self.shifted_sos_tensors: torch.Tensor = self._prepare_shifted_sos_tensors()
    self.shifted_eos_tensors: torch.Tensor = self._prepare_shifted_sos_tensors(use_eos=True)

  def _prepare_idx_shift_by_vocab(self):
    idx_shifts = {}
    idx_range = []
    shift_idx = 1 # 0 is reserved for modality-universal pad token
    for k in self.vocab_keys:
      idx_shifts[k] = shift_idx
      idx_range.append( (shift_idx, shift_idx+len(self.vocabs[k])) )
      shift_idx += len(self.vocabs[k])
    return idx_shifts, idx_range
    
  def _prepare_pos_shift_by_max_seq_len(self):
    pos_shifts = {}
    pos_range = []
    shift_idx = 1 # 0 is reserved for modality-universal pad token
    for k in self.vocab_keys:
      pos_shifts[k] = shift_idx
      if k == 'pt':
        pos_range.append( (shift_idx, shift_idx+self.max_pt_x_len) )
        shift_idx += self.max_pt_x_len
        pos_shifts[k] = (pos_shifts[k], shift_idx)
        pos_range[-1] = (pos_range[-1], (shift_idx, shift_idx + self.max_token_height))
        shift_idx += self.max_token_height
      else:
        pos_range.append( (shift_idx, shift_idx+self.max_seq_len[k]) )
        shift_idx += self.max_seq_len[k]

    return pos_shifts, pos_range, shift_idx

  def _prepare_rq_shifter(self, rvq_vocab:RVQVocab):
    n_codebook = rvq_vocab.n_codebook
    codebook_size = len(rvq_vocab.codebook_vocab)
    shifter = torch.tensor([i*codebook_size for i in range(n_codebook)]).short()
    return shifter
  
  def _prepare_sos_eos_tensors(self):
    sos_tokens = {k: self.vocabs[k].sos_idx for k in self.vocab_keys}
    eos_tokens = {k: self.vocabs[k].eos_idx for k in self.vocab_keys}
    
    # TODO: make this more general
    sos_tensors = {k: torch.tensor([sos_tokens[k]]).short() for k in self.vocab_keys}
    eos_tensors = {k: torch.tensor([eos_tokens[k]]).short() for k in self.vocab_keys}

    if 'dac' in self.vocab_keys:
      sos_tensors['dac'] = sos_tensors['dac'].view(1,1).repeat(1,self.vocabs['dac'].n_codebook)
      eos_tensors['dac'] = eos_tensors['dac'].view(1,1).repeat(1,self.vocabs['dac'].n_codebook)
    
    if 'pt' in self.vocab_keys:
      sos_tensors['pt'] = sos_tensors['pt'].view(1,1).repeat(1,self.vocabs['pt'].n_codebook)
      eos_tensors['pt'] = eos_tensors['pt'].view(1,1).repeat(1,self.vocabs['pt'].n_codebook)

    sos_shifted_tensors = {}
    eos_shifted_tensors = {}
    for k in self.vocab_keys:
      sos_shifted_tensors[k] = sos_tensors[k] + self.idx_shifts[k]
      eos_shifted_tensors[k] = eos_tensors[k] + self.idx_shifts[k]
      if k in self.rq_shifter:
        sos_shifted_tensors[k] = sos_shifted_tensors[k] + self.rq_shifter[k]
        eos_shifted_tensors[k] = eos_shifted_tensors[k] + self.rq_shifter[k]
    
    return sos_tensors, eos_tensors, sos_shifted_tensors, eos_shifted_tensors
  
  def _prepare_logit_mask(self):
    masks = [torch.zeros([self.max_n_codebook, self.vocab_size], dtype=torch.bool) for _ in self.vocab_keys]
    for i, (start, end) in enumerate(self.idx_range):
      if isinstance(self.vocabs[self.vocab_keys[i]], RVQVocab):
        i_vocab_size = self.vocabs[self.vocab_keys[i]].vocab_size # list of vocab size for each codebook
        for j in range(self.vocabs[self.vocab_keys[i]].n_codebook):
          start_idx = start + sum(i_vocab_size[:j])
          end_idx = start_idx + i_vocab_size[j]
          masks[i][j, start_idx:end_idx] = True
      else:
        masks[i][0, start:end] = True
        masks[i][1:, 0] = True # true for pad token to avoid all-zero logits
    if 'pt' in self.vocab_keys:
      masks[self.vocab_keys.index('pt')][:, self.img_crop_cat_sep_idx] = True
      if self.out_pt_height_token:
        for height_token in self.pt_height_tokens:
          masks[self.vocab_keys.index('pt')][:, height_token] = True
    return torch.stack(masks)
  
  def _prepare_shifted_sos_tensors(self, use_eos:bool=False):
    outputs = []
    for k in self.vocab_keys:
      org_tensors = self.sos_tensors if not use_eos else self.eos_tensors
      shifted = org_tensors[k] + self.idx_shifts[k]
      if k in self.rq_shifter:
        shifted = shifted + self.rq_shifter[k]
      if shifted.ndim == 1:
        if use_eos:
          shifted = torch.nn.functional.pad(shifted.unsqueeze(0), (0, self.n_codebook-1), mode='constant', value=self.eos_tensors[k].item())
        else:
          shifted = torch.nn.functional.pad(shifted.unsqueeze(0), (0, self.n_codebook-1), mode='constant', value=self.pad_idx)
      outputs.append(shifted)
    return torch.stack(outputs, dim=0)
  
  def prepare_start_token(self, modal_idx:torch.LongTensor):
    if modal_idx.ndim == 1: # no batch
      out_modal = modal_idx[1].unsqueeze(0)
    else:
      out_modal = modal_idx[:,1]
    return self.shifted_sos_tensors.to(out_modal.device)[out_modal].to(torch.long)
  
  @property
  def vocab_size(self):
    if self.out_pt_height_token:
      return sum([len(self.vocabs[k]) for k in self.vocab_keys]) + 3 + self.max_token_height - 1 # +1 for pad token, +1 for img_crop_cat_sep, + max_token_height - 1 for pt height token
    else:
      return sum([len(self.vocabs[k]) for k in self.vocab_keys]) + 3 # +1 for pad token, +1 for img_crop_cat_sep
  
  @property
  def max_n_codebook(self):
    return max([self.vocabs[k].n_codebook for k in self.vocab_keys])
  
  @property
  def max_tok_len(self):
    # return max([self.max_seq_len[k] if k != 'pt' else self.max_seq_len[k] * self.max_token_height for k in self.vocab_keys])
    return max([self.max_seq_len[k] for k in self.vocab_keys])
  
  def append_sos_eos(self, tokens:torch.Tensor, vocab_key:str, pos:torch.Tensor=None):
    match vocab_key:
      case 'dac':
        tokens = torch.cat([self.sos_shifted_tensors['dac'], tokens, self.eos_shifted_tensors['dac']], dim=0)
        # special_pos = torch.zeros([1]).to(torch.int16)
        # pos = torch.cat([special_pos, pos, special_pos], dim=0).to(torch.int16)
      case 'pt':
        assert pos is not None, 'pos is required for pt'
        x_shift, y_shift, _, n_codebook = tokens.shape
        sos = self.sos_shifted_tensors['pt'].view(1,1,1,-1).repeat(x_shift,y_shift,1,1)
        eos = self.eos_shifted_tensors['pt'].view(1,1,1,-1).repeat(x_shift,y_shift,1,1)
        tokens = torch.cat([sos, tokens, eos], dim=2)
        special_pos = torch.zeros([1,2]).to(torch.int16)
        pos = torch.cat([special_pos,pos,special_pos], dim=0).to(torch.int16)
      case 'midi':
        tokens = torch.cat([self.sos_shifted_tensors['npy'], tokens, self.eos_shifted_tensors['npy']], dim=0)
        # pos = torch.cat([special_pos, pos, special_pos], dim=0).to(torch.int16)
    return tokens

  def make_pos_emb_from_tensor(self, in_batch:torch.Tensor, modal_idx:torch.Tensor, target_height=None):
    assert modal_idx.ndim == 2 # B x 2 
    assert in_batch.ndim == 3 # B x T x N_Codebook
    out_modal = modal_idx[:,1]
    # make lmx pos
    total_pos = torch.zeros([in_batch.shape[0], in_batch.shape[1], 2], dtype=torch.long)
    if in_batch.shape[1] == 1: # first iteration
      return total_pos
    assert max(out_modal) < len(self.vocab_keys)
    for i, vocab_key in enumerate(self.vocab_keys):
      if in_batch[out_modal==i].shape[0] == 0:
        continue
      tokens = in_batch[out_modal==i]
      if vocab_key in ['lmx', 'midi', 'dac']:
        if vocab_key in ['lmx', 'midi']:
          assert (tokens[:, 0, 0].to(self.sos_shifted_tensors[vocab_key].device) == self.sos_shifted_tensors[vocab_key]).all(), f'{vocab_key} sos not at 0th position'
        else:
          assert (tokens[:, 0].to(self.sos_shifted_tensors[vocab_key].device) == self.sos_shifted_tensors[vocab_key]).all(), f'{vocab_key} sos not at 0th position'
        pos = torch.arange(tokens.shape[1], dtype=torch.long) - 1 # Assume sos is at 0th position
        pos += self.pos_shifts[vocab_key]
        pos[0] = 0 # sos is at 0th position
        pos = torch.stack([pos, torch.zeros_like(pos).to(torch.int16)], dim=-1)
        total_pos[out_modal==i] = pos
      elif vocab_key == 'pt':
        if not self.out_pt_height_token:
          # Legacy scheme: padding is appended below the image tokens
          assert target_height is not None, 'target_height is required for pt when out_pt_height_token is False'
          heights = target_height[out_modal==i][:,1]
          assert (heights != 0).all(), 'height should be greater than 0'
          
          sep_points = torch.where(tokens[...,0].to(self.eos_shifted_tensors['pt'].device) == self.img_crop_cat_sep_idx)
          sos_eos_points = torch.where(tokens[...,0].to(self.eos_shifted_tensors['pt'].device) == self.eos_shifted_tensors['pt'][:,0])
          
          pos_x = torch.stack([torch.arange(tokens.shape[1]).to(torch.long)] * tokens.shape[0], dim=0).to(heights.device) - 1
          pos_y = torch.stack([torch.arange(tokens.shape[1]).to(torch.long)] * tokens.shape[0], dim=0).to(heights.device) - 1
          pos_x = pos_x // heights.unsqueeze(1) + self.pos_shifts['pt'][0]
          pos_y = pos_y % heights.unsqueeze(1) + self.pos_shifts['pt'][1]
          pos = torch.stack([pos_x, pos_y], dim=-1)
          pos[:,0,:] = 0  # sos is at 0th position
          
          for row, col in zip(*sep_points):
            # copying pos values from [sep_idx-1:end-1] to [sep_idx:end]
            values_to_copy = pos[row.item(), col.item()-1:-1].clone()
            pos[row.item(), col.item():] = values_to_copy
            
          for row, col in zip(*sos_eos_points):
            if col.item() == 0:  # sos
              pass
            else:
              pos[row.item(), col.item():] = 0  # 0 for eos and after eos
              
        else:
          # New scheme using height tokens
          height_token_points = []
          for ht in self.pt_height_tokens:
            ht_points = torch.where(tokens[...,0].to(self.eos_shifted_tensors['pt'].device) == ht)
            for row, col in zip(*ht_points):
              height_token_points.append((row.item(), col.item(), ht))
          height_token_points.sort(key=lambda x: (x[0], x[1]))  # sort by batch_idx, then col
          
          sep_points = torch.where(tokens[...,0].to(self.eos_shifted_tensors['pt'].device) == self.img_crop_cat_sep_idx)
          sep_points = list(zip(*sep_points))
          sos_eos_points = torch.where(tokens[...,0].to(self.eos_shifted_tensors['pt'].device) == self.eos_shifted_tensors['pt'][:,0])
          sos_eos_points = list(zip(*sos_eos_points))
          
          # Initialize positions
          pos = torch.zeros((tokens.shape[0], tokens.shape[1], 2), dtype=torch.long, device=tokens.device)
          
          # Process each batch element
          max_x_pos = self.pos_shifts['pt'][0] + self.max_pt_x_len - 1  # clamp: degenerate generations must not overflow the position table
          for batch_idx in range(tokens.shape[0]):
            current_x_pos = self.pos_shifts['pt'][0]  # current x position
            current_height = None
            last_content_pos = None
            token_count_in_crop = 0  # token count within the current crop
            
            # Process each column position
            for col in range(tokens.shape[1]):
              # SOS token
              if col == 0:
                pos[batch_idx, col] = 0
                continue
                
              # Check for a height token
              is_height_token = False
              for row, ht_col, ht in height_token_points:
                if row == batch_idx and ht_col == col:
                  # Compute the actual height value of the height token
                  height_idx = self.pt_height_tokens.index(ht)
                  current_height = height_idx + 1
                  token_count_in_crop = 0  # a new crop starts
                  
                  # Height token position: use the same position as the following token
                  pos[batch_idx, col] = torch.tensor([current_x_pos, self.pos_shifts['pt'][1]], 
                                                   dtype=torch.long, device=pos.device)
                  is_height_token = True
                  break
              
              if is_height_token:
                continue
                
              # Check for a separator token
              is_sep = False
              for row, sep_col in sep_points:
                if row == batch_idx and sep_col == col:
                  # The separator keeps the previous content's last position; x_pos advances by 1
                  if last_content_pos is not None:
                    pos[batch_idx, col] = last_content_pos
                    current_x_pos = min(last_content_pos[0].item() + 1, max_x_pos)  # update x position
                  is_sep = True
                  break
              
              if is_sep:
                continue
                
              # Check for the EOS token
              is_eos = False
              for row, eos_col in sos_eos_points:
                if row == batch_idx and eos_col == col and eos_col > 0:  # EOS token (not SOS)
                  pos[batch_idx, col:] = 0  # EOS and everything after it get position 0
                  is_eos = True
                  break
              
              if is_eos:
                break
                
              # Regular content token
              if current_height is not None:
                current_pos_y = self.pos_shifts['pt'][1] + (token_count_in_crop % current_height)
                
                # x position advances by 1 every time y wraps around
                if token_count_in_crop > 0 and token_count_in_crop % current_height == 0:
                    current_x_pos = min(current_x_pos + 1, max_x_pos)
                
                pos[batch_idx, col] = torch.tensor([current_x_pos, current_pos_y], 
                                                 dtype=torch.long, device=pos.device)
                last_content_pos = pos[batch_idx, col].clone()
                token_count_in_crop += 1
        
        total_pos[out_modal==i] = pos.to(total_pos.device)

        # if (total_pos.shape[1] -10) % 50 == 0 or total_pos.shape[1] < 10:
        #   breakpoint()
      else:
        raise ValueError(f'Invalid vocab key: {vocab_key}')
    total_pos[(in_batch==0)[..., 0]] = 0
    return total_pos.to(in_batch.device).to(torch.int16)
  
  def __call__(self, tokens:torch.Tensor, vocab_key:str, append_sos_eos:bool=True, add_height_token:bool=True):
    # add sos and eos tokens
    match vocab_key:
      case 'dac':
        token_height = 0 # dac has no token height
        pos = torch.arange(tokens.shape[1] + 1).to(torch.int16) # +1 for eos
        if append_sos_eos:
          tokens = torch.cat([self.sos_tensors['dac'].unsqueeze(0).repeat(tokens.shape[0],1,1), tokens, self.eos_tensors['dac'].unsqueeze(0).repeat(tokens.shape[0],1,1)], dim=1)
        pos += self.pos_shifts[vocab_key]
        special_pos = torch.zeros([1]).to(torch.int16)
        if append_sos_eos:
          pos = torch.cat([special_pos,pos], dim=0).to(torch.int16) # only add sos as prefix
        pos = torch.stack([pos, torch.zeros_like(pos).to(torch.int16)], dim=-1) # (seq_len, 2) idx[1] is all-zero position; to be compatible with x,y of pt
      case 'pt':
        if isinstance(tokens, list):
          list_of_img_tokens = tokens
          if len(list_of_img_tokens) == 1:
            return self(list_of_img_tokens[0], 'pt', append_sos_eos=append_sos_eos)
          heights = [t.shape[2] for t in list_of_img_tokens]
          widths = [t.shape[3] for t in list_of_img_tokens]
          max_height = max(heights)
          num_pixels = [max_height*w for _, w in zip(heights, widths)]

          for i in range(len(list_of_img_tokens)):
            img = list_of_img_tokens[i]
            height = heights[i]
            width = widths[i]
            pad_tokens = torch.full([img.shape[0], img.shape[1], max_height-height, width, img.shape[4]], self.vocabs['pt'].pad_idx)
            list_of_img_tokens[i] = torch.cat([img, pad_tokens], dim=2)
          
          cat_img = torch.cat(list_of_img_tokens, dim=3)
          tokens, token_height, img_pos = self(cat_img, 'pt', add_height_token=False)
          # add sep tokens
          sep_tokens = torch.full([tokens.shape[0], tokens.shape[1], 1, tokens.shape[3]], self.img_crop_cat_sep_idx)
          tokens_separated = []
          pos_separated = []
          cur_idx = 0
          num_pixels[0] += 1 # add one for the sos token
          num_pixels[-1] += 1
          
          # Prepare height tokens if needed
          if self.out_pt_height_token:
            height_tokens = []
            for height in heights:
              # Get the appropriate height token (height=1 at index 0, height=2 at index 1, etc.)
              height_idx = min(height - 1, len(self.pt_height_tokens) - 1)  # Ensure we don't go out of bounds
              height_token = torch.full([tokens.shape[0], tokens.shape[1], 1, tokens.shape[3]], 
                                       self.pt_height_tokens[height_idx])
              height_tokens.append(height_token)
          
          for i in range(len(num_pixels)):
            # Add current image segment tokens
            curr_tokens = tokens[:, :, cur_idx:cur_idx+num_pixels[i]]
            curr_pos = img_pos[cur_idx:cur_idx+num_pixels[i]]
            
            # If using height tokens, insert the height token after SOS or SEP
            if self.out_pt_height_token:
              if i == 0:
                # First segment: insert height token between SOS and image content
                sos_token = curr_tokens[:, :, 0:1, :]
                img_tokens = curr_tokens[:, :, 1:, :]
                sos_pos = curr_pos[0:1]
                img_pos_part = curr_pos[1:]
                
                tokens_separated.append(sos_token)  # SOS
                tokens_separated.append(height_tokens[i])  # Height token
                tokens_separated.append(img_tokens)  # Image tokens
                
                pos_separated.append(sos_pos)
                pos_separated.append(img_pos_part[0:1])  # Position for height token
                pos_separated.append(img_pos_part)
              else:
                # Non-first segments: just add all tokens (height was already added after SEP)
                tokens_separated.append(curr_tokens)
                pos_separated.append(curr_pos)
            else:
                tokens_separated.append(curr_tokens)
                pos_separated.append(curr_pos)
            
            cur_idx += num_pixels[i]
            
            # Add separator after all but the last image
            if i < len(num_pixels)-1:
              tokens_separated.append(sep_tokens)
              # Use the position of the last token from current image segment for separator
              sep_pos = img_pos[cur_idx-1:cur_idx]
              pos_separated.append(sep_pos)
              
              # Add height token after separator (for the next image)
              if self.out_pt_height_token:
                tokens_separated.append(height_tokens[i+1])
                # Get the position for the height token (use first token of next segment)
                next_pos = img_pos[cur_idx:cur_idx+1] if cur_idx < len(img_pos) else sep_pos
                pos_separated.append(next_pos)
          
          tokens_separated = torch.cat(tokens_separated, dim=2)
          pos_separated = torch.cat(pos_separated, dim=0)
          
          # Remove padding tokens if height tokens are used
          if self.out_pt_height_token:
            # Find padding token positions - uses shifted indices
            pad_mask = tokens_separated[..., 0] != (self.vocabs['pt'].pad_idx + self.idx_shifts['pt'])
            
            # SOS, EOS, SEP, and height tokens must also be preserved - all use shifted indices
            shifted_sos = self.sos_tensors['pt'][0, 0] + self.idx_shifts['pt']
            shifted_eos = self.eos_tensors['pt'][0, 0] + self.idx_shifts['pt']
            
            special_tokens_mask = (tokens_separated[..., 0] == shifted_sos) | \
                                 (tokens_separated[..., 0] == shifted_eos) | \
                                 (tokens_separated[..., 0] == self.img_crop_cat_sep_idx)
            
            # Preserve height tokens too (every token in pt_height_tokens)
            for ht in self.pt_height_tokens:
                special_tokens_mask = special_tokens_mask | (tokens_separated[..., 0] == ht)
            
            # Combine masks: keep positions that are non-padding or special tokens
            keep_mask = pad_mask | special_tokens_mask
            
            # Filter tokens and position info
            x_shift, y_shift, seq_len, n_codebook = tokens_separated.shape
            
            # Lists holding the filtered tokens of each batch element
            filtered_tokens_list = []
            filtered_pos_list = []
            
            # Compute the maximum filtered length
            max_filtered_len = 0
            for x in range(x_shift):
                for y in range(y_shift):
                    curr_mask = keep_mask[x, y]
                    if curr_mask.any():
                        filtered_len = curr_mask.sum().item()
                        max_filtered_len = max(max_filtered_len, filtered_len)
            
            # Prepare new tensors (same maximum length for every batch element)
            tokens_final = torch.full([x_shift, y_shift, max_filtered_len, n_codebook], 
                                     self.vocabs['pt'].pad_idx + self.idx_shifts['pt'],
                                     device=tokens_separated.device, 
                                     dtype=tokens_separated.dtype)
            
            # Assign the filtered tokens of each batch element
            for x in range(x_shift):
                for y in range(y_shift):
                    curr_mask = keep_mask[x, y]
                    if curr_mask.any():
                        filtered_tokens = tokens_separated[x, y, curr_mask]
                        filtered_len = filtered_tokens.shape[0]
                        tokens_final[x, y, :filtered_len] = filtered_tokens
            
            # Filter position info
            # Every batch element shares the same position info, so pick one
            for x in range(x_shift):
                for y in range(y_shift):
                    curr_mask = keep_mask[x, y]
                    if curr_mask.any():
                        filtered_pos = pos_separated[curr_mask]
                        # Pad if necessary
                        if filtered_pos.shape[0] < max_filtered_len:
                            pad_pos = torch.zeros([max_filtered_len - filtered_pos.shape[0], 2], 
                                                 device=filtered_pos.device, 
                                                 dtype=filtered_pos.dtype)
                            filtered_pos_padded = torch.cat([filtered_pos, pad_pos], dim=0)
                        else:
                            filtered_pos_padded = filtered_pos
                        return tokens_final, max_height, filtered_pos_padded
            
            # On filtering failure, return the originals
            return tokens_separated, max_height, pos_separated
            
          return tokens_separated, max_height, pos_separated
        match tokens.ndim:
          case 5:
            token_height = tokens.shape[2] # pt has token height (2D Image)
            tokens = tokens.transpose(2,3) # (x_shift, y_shift, h, w, n_tokens) -> (x_shift, y_shift, w, h, n_tokens)
            
            pos_x = torch.arange(tokens.shape[2]).to(torch.int16) 
            pos_x += self.pos_shifts[vocab_key][0]
            pos_x = pos_x.unsqueeze(-1).repeat(1,tokens.shape[3]) # (w) -> (w, h)
            pos_y = torch.arange(tokens.shape[3]).to(torch.int16) 
            pos_y += self.pos_shifts[vocab_key][1]
            pos_y = pos_y.unsqueeze(0).repeat(tokens.shape[2],1) # (h) -> (w, h)
            pos = torch.stack([pos_x, pos_y], dim=-1) # (w, h, 2) 2 for x, y

            pos = pos.flatten(0,1) # (w, h, 2) -> (w*h, 2) 2 for x, y
            tokens = tokens.flatten(2,3) # (x_shift, y_shift, w, h, n_tokens) -> (x_shift, y_shift, w*h, n_tokens)
            
            x_shift, y_shift, _, n_codebook = tokens.shape
            if append_sos_eos:
              sos = self.sos_tensors['pt'].view(1,1,1,-1).repeat(x_shift,y_shift,1,1)
              eos = self.eos_tensors['pt'].view(1,1,1,-1).repeat(x_shift,y_shift,1,1)
              
              # Create height token if needed
              if self.out_pt_height_token and add_height_token:
                # Get the correct height token based on original token height
                height_idx = min(token_height - 1, len(self.pt_height_tokens) - 1)
                height_token = torch.full([x_shift, y_shift, 1, n_codebook], 
                                         self.pt_height_tokens[height_idx] - self.idx_shifts['pt']) - self.rq_shifter['pt'] # will be shifted by idx_shifts['pt'] and rq_shifter['pt'] later
                # Insert SOS, height token, and then the image tokens
                tokens = torch.cat([sos, height_token, tokens, eos], dim=2)
                
                # Position info for height token (use the first token's position)
                sos_pos = torch.zeros([1,2]).to(torch.int16)
                first_token_pos = pos[0:1].clone() if len(pos) > 0 else sos_pos
                eos_pos = torch.zeros([1,2]).to(torch.int16)
                eos_pos[0,0] = pos_x.max() + 1
                eos_pos[0,1] = self.pos_shifts[vocab_key][1]
                pos = torch.cat([sos_pos, first_token_pos, pos, eos_pos], dim=0).to(torch.int16)
              else:
                # Original behavior without height token
                tokens = torch.cat([sos, tokens, eos], dim=2)
                sos_pos = torch.zeros([1,2]).to(torch.int16)
                eos_pos = torch.zeros([1,2]).to(torch.int16)
                eos_pos[0,0] = pos_x.max() + 1
                eos_pos[0,1] = self.pos_shifts[vocab_key][1]
                pos = torch.cat([sos_pos, pos, eos_pos], dim=0).to(torch.int16)
          case 6:
            token_height = tokens.shape[3] # pt has token height (2D Image)
            tokens = tokens.transpose(3,4) # (aug, x_shift, y_shift, h, w, n_tokens) -> (aug, x_shift, y_shift, w, h, n_tokens)
            
            pos_x = torch.arange(tokens.shape[3]).to(torch.int16) 
            pos_x += self.pos_shifts[vocab_key][0]
            pos_x = pos_x.unsqueeze(-1).repeat(1,tokens.shape[4]) # (w) -> (w, h)
            pos_y = torch.arange(tokens.shape[4]).to(torch.int16) 
            pos_y += self.pos_shifts[vocab_key][1]
            pos_y = pos_y.unsqueeze(0).repeat(tokens.shape[3],1) # (h) -> (w, h)
            pos = torch.stack([pos_x, pos_y], dim=-1) # (w, h, 2) 2 for x, y

            pos = pos.flatten(0,1) # (w, h, 2) -> (w*h, 2) 2 for x, y
            tokens = tokens.flatten(3,4) # (aug, x_shift, y_shift, w, h, n_tokens) -> (aug, x_shift, y_shift, w*h, n_tokens)
            
            aug_idx, x_shift, y_shift, _, n_codebook = tokens.shape
            if append_sos_eos:
              sos = self.sos_tensors['pt'].view(1,1,1,1,-1).repeat(aug_idx,x_shift,y_shift,1,1)
              eos = self.eos_tensors['pt'].view(1,1,1,1,-1).repeat(aug_idx,x_shift,y_shift,1,1)
              
              # Create height token if needed
              if self.out_pt_height_token and add_height_token:
                # Get the correct height token based on original token height
                height_idx = min(token_height - 1, len(self.pt_height_tokens) - 1)
                height_token = torch.full([aug_idx, x_shift, y_shift, 1, n_codebook], 
                                         self.pt_height_tokens[height_idx] - self.idx_shifts['pt']) - self.rq_shifter['pt'] # will be shifted by idx_shifts['pt'] and rq_shifter['pt'] later
                # Insert SOS, height token, and then the image tokens
                tokens = torch.cat([sos, height_token, tokens, eos], dim=3)
                
                # Position info for height token (use the first token's position)
                sos_pos = torch.zeros([1,2]).to(torch.int16)
                first_token_pos = pos[0:1].clone() if len(pos) > 0 else sos_pos
                eos_pos = torch.zeros([1,2]).to(torch.int16)
                eos_pos[0,0] = pos_x.max() + 1
                eos_pos[0,1] = self.pos_shifts[vocab_key][1]
                pos = torch.cat([sos_pos, first_token_pos, pos, eos_pos], dim=0).to(torch.int16)
              else:
                # Original behavior without height token
                tokens = torch.cat([sos, tokens, eos], dim=3)
                sos_pos = torch.zeros([1,2]).to(torch.int16)
                eos_pos = torch.zeros([1,2]).to(torch.int16)
                eos_pos[0,0] = pos_x.max() + 1
                eos_pos[0,1] = self.pos_shifts[vocab_key][1]
                pos = torch.cat([sos_pos, pos, eos_pos], dim=0).to(torch.int16)
          case _:
            raise ValueError(f'Invalid tensor shape for image tokens: {tokens.shape}')
      case 'lmx':
        # tokens is a string of lmx
        tokens = self.vocabs['lmx'](tokens)
        tokens = torch.tensor(tokens, dtype=torch.short)
        token_height = 0 # lmx has no token height
        tokens = tokens.unsqueeze(1) # already has sos and eos tokens. make rvq compatible
        pos = torch.arange(tokens.shape[0]).to(torch.int16) -1 # already has sos in 0th position so 1th position should be 0
        pos += self.pos_shifts[vocab_key]
        pos[0] = 0 # sos
        pos = torch.stack([pos, torch.zeros_like(pos).to(torch.int16)], dim=-1) # (seq_len, 2) idx[1] is all-zero position; to be compatible with x,y of pt
      case 'midi':
        token_height = 0 # npy has no token height
        tokens = torch.cat([self.sos_tensors['midi'], torch.Tensor(tokens).to(torch.int16), self.eos_tensors['midi']], dim=0)
        tokens = tokens.unsqueeze(1)
        pos = torch.arange(tokens.shape[0]).to(torch.int16) - 1 # already has sos in 0th position so 1th position should be 0
        pos += self.pos_shifts[vocab_key]
        pos[0] = 0 # sos
        pos = torch.stack([pos, torch.zeros_like(pos).to(torch.int16)], dim=-1) # (seq_len, 2) idx[1] is all-zero position; to be compatible with x,y of pt

    # Shift the indices by vocab order
    tokens = tokens + self.idx_shifts[vocab_key]
    
    # Shift the indices by codebook order
    if vocab_key in self.rq_shifter:
      tokens = tokens + self.rq_shifter[vocab_key]
    
    return tokens, token_height, pos