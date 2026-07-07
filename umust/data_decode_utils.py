import numpy as np
import torch
import cv2
import os
import shutil
import wandb

from collections import defaultdict

from pydub import AudioSegment
from pathlib import Path
from .vocab_utils import VQVocab, RVQVocab
from .lmx_utils import delinearize_lmx, render_xml_with_musescore
from .midi_utils.midi import note_event2midi
from .utils import load_vq_model_mm, get_fluidsynth
from .dac_utils import LSDAC
from tqdm.auto import tqdm
from .midi_utils.event2note import merge_zipped_note_events_and_ties_to_notes
from .midi_utils.note2event import note2note_event
from .evaluation_utils import draw_attention_map

def getitem_from_batch(batch, idx):
  return {key: val[idx] for key, val in batch.items()}


class TensorDecoder:
  def __init__(self, config, in_vocab, out_vocab, out_dir:Path, device='cpu', vq_model=None, dac_model=None):
    self.config = config
    self.in_vocab = in_vocab
    self.out_vocab = out_vocab
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.device = device
    
    
    if vq_model is not None:
      self.vq_model = vq_model
      self.vq_model.to(device)
      self.vq_model.eval()
    elif config.data.vq_model is not None:
      self.vq_model = load_vq_model_mm(config)
      self.vq_model.to(device)
      self.vq_model.eval()
    if dac_model is not None:
      self.dac_model = dac_model
      self.dac_model.to(device)
      self.dac_model.eval()
    elif config.data.dac_model is not None:
      dac_model_dir = Path('dac_models') / config.data.dac_model 
      self.dac_model = LSDAC.load( dac_model_dir / 'weights.pth' )
      self.dac_model.to(device)
      self.dac_model.eval()
    if 'midi' in config.data.out_modal_type or 'midi' in config.data.in_modal_type:
      self.fs = get_fluidsynth()
    
    
    self.modal_vocab = {}
    for modal_str, vocab in self.in_vocab.vocabs.items():
      self.modal_vocab[modal_str] = vocab
    for modal_str, vocab in self.out_vocab.vocabs.items():
      self.modal_vocab[modal_str] = vocab
    
    

  def truncate_output(self, inferenced_output, modality, vocab):
    eos = vocab.shifted_eos_tensors[modality].to(inferenced_output.device) # tensor
    if vocab.vocab_keys[modality] in ['lmx', 'midi']:
      eos = eos[0,0] # integer
    if (inferenced_output[:, 0] == eos).any():
      inferenced_output = inferenced_output[:, 1:]
    # Find first occurrence of EOS token along sequence dimension
    eos_mask = (inferenced_output.to(eos.device) == eos).any(dim=-1) # It is faster than .all() and also works for partial eos token tensors
    if eos_mask.any():
      eos_idx = eos_mask.float().argmax(dim=1)
      inferenced_output = inferenced_output[:, :eos_idx]
    return inferenced_output

  def decode_dac(self, inferenced_output, filename, custom_fn=False):
    self.dac_model.to(self.device)
    inferenced_output = inferenced_output.permute(0, 2, 1)
    signal = self.dac_model.decompress_tensor(
      inferenced_output.to(self.device), 
      n_quantizers=self.config.data.n_codebook, 
    )
    signal = signal.to('cpu')
    # save decoded audio
    if custom_fn:
      out_path = Path(f"{str(filename)}:decoded_audio.wav")
    else:
      out_path = self.out_dir / f"{filename}:decoded_audio.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signal.write(out_path)    
    AudioSegment.from_wav(out_path).export(out_path.with_suffix(".mp3"), format="mp3")
    # os.remove(out_path)
    return inferenced_output, (str(out_path.with_suffix(".mp3")), )

  def decode_lmx(self, inferenced_output, vocab):
    inferenced_output = inferenced_output.to('cpu').squeeze(0)
    assert inferenced_output.ndim == 1
    lmx_str = self.modal_vocab['lmx'].decode(inferenced_output)
    
    xml_str = delinearize_lmx(lmx_str)
    try:
      rendered_img = render_xml_with_musescore(xml_str)
    except Exception as e:
      print(f"Score rendering failed ({e}); logging LMX text only")
      rendered_img = None

    return inferenced_output, lmx_str, rendered_img

  def decode_midi(self, inferenced_output, filename, custom_fn=False):
    inferenced_output = inferenced_output.squeeze(0).cpu().tolist()
    decoded_events = self.modal_vocab['midi'].decode(inferenced_output)
    raw_events = [self.modal_vocab['midi'].codec.decode_event_index(x) for x in inferenced_output]
    raw_events_str = ' '.join([f"{e.type}:{e.value}" for e in raw_events])
    note_events, tie_note_events, last_activity, err_cnt = decoded_events
    if custom_fn:
      out_path = Path(f"{str(filename)}:decoded_midi.mid")
    else:
      out_path = self.out_dir / f'{filename}:decoded_midi.mid'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    note_event2midi(note_events, output_file=out_path)
    if shutil.which('fluidsynth') is not None:
      self.fs.midi_to_audio(out_path, out_path.with_suffix('.wav'))
      AudioSegment.from_wav(out_path.with_suffix('.wav')).export(out_path.with_suffix(".mp3"), format="mp3")
      os.remove(out_path.with_suffix('.wav'))
    else:
      print("fluidsynth not found; skipping audio rendering of decoded MIDI")
    str_out_path = out_path.with_suffix(".txt")
    with open(str_out_path, 'w') as f:
      f.write(raw_events_str)
    return inferenced_output, (str(out_path), str(out_path.with_suffix(".mp3")), str(str_out_path))
  
  def decode_piece_midi(self, inferenced_output:list[torch.Tensor], hop_len=10, filename=None, custom_fn=False):
    list_batch_tokens = []
    list_start_times = []
    cur_time = 0

    for segment in inferenced_output:
      in_tensor, _ = self.unshift_tensor(segment, torch.tensor(self.out_vocab.vocab_keys.index('midi')), is_in=False)
      list_batch_tokens.append(in_tensor.numpy())
      list_start_times.append(cur_time)
      cur_time +=  hop_len

    zipped_note_events_and_tie, list_events, ne_err_cnt = self.modal_vocab['midi'].decode_list_batches(list_batch_tokens, list_start_times, return_events=True)
    pred_notes, n_err_cnt = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
    if filename is not None:
      pred_events = note2note_event(pred_notes)
      if custom_fn:
        out_path = f"{filename}:decoded_midi.mid"
      else:
        out_path = self.out_dir / f'{filename}:decoded_midi.mid'
      out_path.parent.mkdir(parents=True, exist_ok=True)
      note_event2midi(pred_events, output_file=out_path)
      if shutil.which('fluidsynth') is not None:
        self.fs.midi_to_audio(out_path, out_path.with_suffix('.wav'))
        AudioSegment.from_wav(out_path.with_suffix('.wav')).export(out_path.with_suffix(".mp3"), format="mp3")
        os.remove(out_path.with_suffix('.wav'))
      else:
        print("fluidsynth not found; skipping audio rendering of decoded MIDI")
      return pred_notes, (str(out_path), str(out_path.with_suffix(".mp3")))    
    else:
      return pred_notes


  def decode_pt(self, inferenced_output, height, filename, use_in:bool=True, custom_fn=False):
    self.vq_model.to(self.device)
    vocab = self.in_vocab if use_in else self.out_vocab
    
    sep_token = vocab.img_crop_cat_sep_idx
    # Since all the tokens are shifted by idx_shifts['pt'] and rq_shifter['pt'][0], we need to shift the sep_token also
    sep_token = sep_token - vocab.idx_shifts['pt']
    # Finding the sep_token for each codebook
    sep_token_0 = sep_token - vocab.rq_shifter['pt'][0]
    sep_token_1 = sep_token - vocab.rq_shifter['pt'][1]
    sep_token_2 = sep_token - vocab.rq_shifter['pt'][2]
    sep_token_3 = sep_token - vocab.rq_shifter['pt'][3]

    # find sep_token_idx for each codebook
    sep_token_idxs_0 = (inferenced_output[...,0] == sep_token_0).nonzero(as_tuple=True)[1]
    sep_token_idxs_1 = (inferenced_output[...,1] == sep_token_1).nonzero(as_tuple=True)[1]
    sep_token_idxs_2 = (inferenced_output[...,2] == sep_token_2).nonzero(as_tuple=True)[1]
    sep_token_idxs_3 = (inferenced_output[...,3] == sep_token_3).nonzero(as_tuple=True)[1]
    
    # Combine all indices
    sep_token_idxs = torch.cat([sep_token_idxs_0, sep_token_idxs_1, sep_token_idxs_2, sep_token_idxs_3])
    sep_token_idxs = torch.unique(sep_token_idxs)  # Remove duplicates and sort

    if sep_token_idxs.numel() > 0:
    
      # Split the output at every sep token position
      sections = []
      start_idx = 0
      
      for sep_idx in sep_token_idxs:
        # Add the section up to (but excluding) the sep token
        section = inferenced_output[:, start_idx:sep_idx]
        sections.append(section)
        start_idx = sep_idx + 1  # resume right after the sep token
      
      # Also add the section after the last sep token
      if start_idx < inferenced_output.shape[1]:
        sections.append(inferenced_output[:, start_idx:])
    else:
      sections = [inferenced_output]

    imgs = []
    # print(f"Decoding {len(sections)} sections")
    
    for s, section in enumerate(sections):
      if not use_in and self.out_vocab.out_pt_height_token:
        # Find the height token index in the pt_height_tokens list
        height_token_value = section[0,0,0].item()
        assert height_token_value in self.out_vocab.pt_height_tokens, f"Height token value {height_token_value} not in pt_height_tokens"
        height_idx = self.out_vocab.pt_height_tokens.index(height_token_value) + 1
        height = height_idx + 1  # Convert index to actual height value (index 0 = height 1)

        # Skip the first token (height token)
        section = section[:, 1:]

      section = section[:,:section.shape[1] // height * height] # truncate to the nearest multiple of height
      if section.shape[1] == 0:
        print(f"Section {s} is empty")
        continue
      section = section.reshape(section.shape[0], -1, height, section.shape[-1]).transpose(-3,-2)

      image = self.vq_model.decode_code(section.to(self.device))
      image = image * 255 # RQVAE outputs are in range [0,1]
      image = image.squeeze(0).permute(1,2,0).cpu().numpy()
      imgs.append(image)
      # save decoded image
      if custom_fn:
        img_path = Path(str(filename) + f":decoded_pt_image:{s}.png")
      else:
        img_path = self.out_dir / f"{filename}:decoded_pt_image:{s}.png"
      img_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(img_path), image)
      
      sections[s] = section

    if len(imgs) == 0:
      raise Exception(f"No images decoded")

    # Concatenate all images horizontally
    if len(imgs) > 1:
      heights = [img.shape[0] for img in imgs]
      if len(set(heights)) > 1:
        # Pad images to the maximum height with white pixels
        max_height = max(heights)
        for i in range(len(imgs)):
          if imgs[i].shape[0] < max_height:
            # Create white padding (255 for all channels)
            padding_height = max_height - imgs[i].shape[0]
            padding = np.ones((padding_height, imgs[i].shape[1], imgs[i].shape[2]), dtype=imgs[i].dtype) * 255
            # Concatenate the padding at the bottom of the image
            imgs[i] = np.concatenate([imgs[i], padding], axis=0)
      image = np.concatenate(imgs, axis=1)
      # Save concatenated image
      if custom_fn:
        img_path = Path(str(filename) + f":decoded_pt_image_concat.png")
      else:
        img_path = self.out_dir / f"{filename}:decoded_pt_image_concat.png"
      img_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(img_path), image)
    
    return sections, (str(img_path),)
  
  def unshift_tensor(self, in_tensor, modal_idx, is_in:bool=True):
    assert modal_idx.ndim == 0
    vocab = self.in_vocab if is_in else self.out_vocab
    in_modal_str = vocab.vocab_keys[modal_idx.item()]
    # Log inputs by modality
    if in_tensor.ndim <= 2:
      in_tensor = in_tensor.unsqueeze(0)
    in_tensor = self.truncate_output(in_tensor, modal_idx.item(), vocab)

    # Shift back the indices by vocab order
    in_tensor = in_tensor - vocab.idx_shifts[in_modal_str]
    # Shift back the indices by codebook order
    if in_modal_str in vocab.rq_shifter:
      in_tensor = in_tensor - vocab.rq_shifter[in_modal_str].to(in_tensor.device)
    else:
      in_tensor = in_tensor
    in_tensor[in_tensor<0] = 0
    if in_modal_str in ['midi', 'lmx']:
      in_tensor = in_tensor[..., 0]
    
    return in_tensor, in_modal_str
  
  def draw_attention_map(self, attn_weights, dataset_name, n_iter=0, custom_idx=None, custom_sub_idx=None):
    attn_paths = []
    for j in range(attn_weights.shape[1]):
      attn = attn_weights[:,j]
      attention_map = draw_attention_map(attn)
      data_id = str(custom_idx[j]) if custom_idx is not None else str(j)
      data_id = str(custom_idx[j]) + '/' + str(custom_sub_idx[j]) if custom_sub_idx is not None else data_id
      attention_map_img_path = self.out_dir / f"{dataset_name}/{data_id}:{n_iter}:attn_map.png"
      attention_map_img_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(attention_map_img_path, attention_map)
      attn_paths.append(str(attention_map_img_path))
    return attn_paths

  def decode_tensor_by_modality(self, 
                                atensor, 
                                modal_str:str, 
                                dataset_name:str='', 
                                data_type:str='', 
                                data_id:str='0', 
                                token_heights:torch.Tensor=None, 
                                use_in:bool=True,
                                n_iter:int=0,
                                custom_output_fn:Path=None):
    
    output_fn = f"{dataset_name}/{data_id}:{n_iter}_{data_type}"
    if custom_output_fn is not None:
      output_fn = custom_output_fn
      custom_fn = True
    else:
      custom_fn = False
    match modal_str:
      case 'lmx':
        _, lmx_str, rendered_img = self.decode_lmx(atensor, self.in_vocab)
        decoded_file_fn = []
        if rendered_img is not None:
          rendered_img_path = self.out_dir / f"{output_fn}_lmx_image.png"
          if custom_output_fn is not None:
            rendered_img_path = Path(str(custom_output_fn) + '_lmx_image.png')
          rendered_img_path.parent.mkdir(parents=True, exist_ok=True)
          cv2.imwrite(rendered_img_path, rendered_img)
          decoded_file_fn.append(str(rendered_img_path))
        if lmx_str is not None:
          # Save lmx string to file
          lmx_filepath = self.out_dir / f"{output_fn}_input.lmx"
          if custom_output_fn is not None:
            lmx_filepath = Path(str(custom_output_fn) + '_input.lmx')
          lmx_filepath.parent.mkdir(parents=True, exist_ok=True)
          with open(lmx_filepath, 'w') as f:
            f.write(lmx_str)
          decoded_file_fn.append(str(lmx_filepath))
        decoded_file_fn = tuple(decoded_file_fn)
      case 'midi':
        _, decoded_file_fn = self.decode_midi(atensor.squeeze(), output_fn, custom_fn)
      case 'pt':
        height = token_heights.item()
        assert height != 0, f"Height is 0: {height}"
        _, decoded_file_fn = self.decode_pt(atensor, height, output_fn, use_in=use_in, custom_fn=custom_fn)
      case 'dac':
        _, decoded_file_fn = self.decode_dac(atensor, output_fn, custom_fn)
      case _:
        raise Exception(f"Invalid in modality: {modal_str}")
    
    decoded_file_fn = tuple([x for x in decoded_file_fn if Path(x).exists()])
    return decoded_file_fn
  
  
  def make_log_dict(self, modal_idx, dataset_names, decoded_file_fns, data_type:str, log_dict=None, n_iter:int=0):
    if log_dict is None:
      log_dict = defaultdict(list)
    log_keys = []
    ext2log_key = {
      'png': 'image',
      'mp3': 'audio',
      'lmx': 'lmx',
      'mid': 'midi',
    }
    for (in_modal_idx, out_modal_idx), dataset_name in zip(modal_idx, dataset_names):
      in_modal_str = self.in_vocab.vocab_keys[in_modal_idx]
      out_modal_str = self.out_vocab.vocab_keys[out_modal_idx]
      log_keys.append(f"{in_modal_str}2{out_modal_str}:{dataset_name}")
    for i, fns in enumerate(decoded_file_fns):
      for fn in fns:
        file_ext = fn.split('.')[-1]
        sample_key = log_keys[i] + '/' + f"{data_type}_{ext2log_key[file_ext]}"
        match file_ext:
          case 'png':
            log_dict[sample_key].append(wandb.Image(str(fn), caption=f"{data_type}:{n_iter}_{ext2log_key[file_ext]}"))
          case 'mp3':
            log_dict[sample_key].append(wandb.Audio(str(fn), caption=f"{data_type}:{n_iter}_{ext2log_key[file_ext]}"))
          case _:
            # print(f"Invalid file extension: {file_ext}")
            continue
    return log_dict

  def __call__(self, batch_tensor:torch.Tensor, modal_idx:torch.Tensor, dataset_names:list[str], data_type:str='', token_heights:torch.Tensor=None, use_in:bool=True, n_iter:int=0, custom_idx:list[int]=None, custom_sub_idx:list[int]=None, custom_output_fns:list[str]=None):
    if custom_idx is not None:
      assert len(custom_idx) == len(batch_tensor)
      assert len(custom_sub_idx) == len(batch_tensor)
      
    total_decoded_file_fns = []
    if isinstance(dataset_names, str):
      dataset_names = [dataset_names] * len(batch_tensor)
    assert len(dataset_names) == len(batch_tensor)
    for i, (sample_i, sample_modal_idx, sample_dataset_name) in tqdm(enumerate(zip(batch_tensor, modal_idx, dataset_names)), desc="Decoding"):
      in_tensor, in_modal_str = self.unshift_tensor(sample_i.to(self.device), sample_modal_idx.to(self.device), use_in)
      i_token_height = token_heights[i,1-use_in] if token_heights is not None else None
      # try:
      data_id = str(custom_idx[i]) if custom_idx is not None else str(i)
      data_id = str(custom_idx[i]) + '/' + str(custom_sub_idx[i]) if custom_sub_idx is not None else data_id
      decoded_file_fns = self.decode_tensor_by_modality(in_tensor, in_modal_str, sample_dataset_name, data_type, data_id, i_token_height, use_in=use_in, n_iter=n_iter, custom_output_fn=custom_output_fns[i] if custom_output_fns is not None else None)
      # except Exception as e:
      #   print(e)
      #   print(f"Failed to decode {sample_dataset_name} {in_modal_str} {data_type} {i}")
      #   decoded_file_fns = []
      total_decoded_file_fns.append(decoded_file_fns)
    return total_decoded_file_fns
  
  
  # def decode_batch_result(self, batch_tensor, modal_idx, dataset_names, log_input_and_gt:bool=True):
  #   for i, (sample_i, sample_modal_idx, sample_dataset_name) in tqdm(enumerate(zip(batch_tensor, modal_idx, dataset_names)), desc="Decoding"):
      
  #     in_tensor, in_modal_str = self.unshift_tensor(sample_i, sample_modal_idx, use_in)
  #     i_token_height = token_heights[i,0] if token_heights is not None else None
  #     out = self.decode_tensor_by_modality(in_tensor, in_modal_str, dataset_names, i, i_token_height, use_in=use_in)
  #   return out