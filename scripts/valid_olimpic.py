import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from umust.utils import get_model, convert_wandb_style_config_to_omega_config, get_vq_model, get_dac_model, get_dataset, load_model_state_dict
from umust.data_utils import MultimodalTokenDatasetMaker
from umust.model_zoo import MultimodalTranslator
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='OMR validation (SER) on the OLiMPiC validation set')
parser.add_argument('--run_path', type=Path, required=True, help='training run directory containing files/config.yaml and files/checkpoints/')
parser.add_argument('--iteration', type=int, default=None, help='checkpoint iteration to load (default: latest)')
parser.add_argument('--data_dir', type=str, default='dataset/', help='root directory of the preprocessed datasets')
args = parser.parse_args()

run_path = args.run_path
iteration = args.iteration

config_path = run_path / 'files' / 'config.yaml'
config = OmegaConf.load(config_path)
try:
  config = convert_wandb_style_config_to_omega_config(config)
except:
  pass
config.data.vq_model = 'unirqvae3'
config.data.dac_model = 'unidac4'
data_config = config.data

# config.data.lmx_vocab_path = 'vocab/lmx_vocab_singletoken.txt'


data_config.data_path = [x for x in data_config.data_path if 'olimpic'  in x[0]]


model_ckpt_dir = run_path / 'files' / 'checkpoints'
pt_fns = list(model_ckpt_dir.glob('*.pt'))

if iteration:
  model_path = Path(next(p for p in pt_fns if str(iteration) in p.stem))
else:
  sorted_pt_fns = sorted(pt_fns, key=lambda x: int(x.stem.split('_')[0].replace('iter', '')))
  model_path = sorted_pt_fns[-1]

print(model_path)

data_config.data_dir = args.data_dir
# dataset = MultimodalTokenDatasetMaker(
#   data_path = data_config.data_path,
#   data_dir = data_config.data_dir,
#   metadata_dir = Path.cwd() / data_config.metadata_dir,
#   n_codebook = data_config.n_codebook,
#   codebook_size = data_config.codebook_size,
#   max_seq_len = data_config.max_seq_len,
#   num_special_tokens = data_config.num_special_tokens,
#   image_height = data_config.image_height, # This must be the max image height in the dataset
#   image_compress_factor = data_config.image_compress_factor,
#   midi_max_shift = data_config.midi_max_shift,
#   in_modal_type = data_config.in_modal_type,
#   out_modal_type = data_config.out_modal_type,
#   debug = config.general.debug,
# )
dataset = get_dataset(config)

in_idx_handler, out_idx_handler = dataset.in_idx_handler, dataset.out_idx_handler

vq_model, vq_emb = get_vq_model(config)
dac_model, dac_emb = get_dac_model(config)
# model = get_model(config, vq_emb, dac_emb, dataset)

model = MultimodalTranslator(config.nn_params, dataset.in_idx_handler, dataset.out_idx_handler)

state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']
load_model_state_dict(model, state_dict)
print("Loaded model state dict")
model.eval()
model.cuda()


scanned_data_pairs = [x for x in dataset.valid_data_pairs['olimpic'] if 'scanned' in str(x['lmx'])]
synthetic_data_pairs = [x for x in dataset.valid_data_pairs['olimpic'] if 'synthetic' in str(x['lmx'])]

dataset.valid_data_pairs = {'olimpic-scanned': scanned_data_pairs, 'olimpic-synthetic': synthetic_data_pairs}

olimpic_loader = dataset.get_specific_testset_loader('olimpic-scanned', 'pt', 'lmx', batch_size=32, use_valid=True)


total_inf_out = []
total_target_out = []
i = 0
with torch.inference_mode():
  for batch in tqdm(olimpic_loader):
    in_modal, in_mask, target_in, target_out, modal_idx, token_heights, in_pos, target_in_pos = batch['in_modal'], batch['in_mask'], batch['target_in'], batch['target_out'], batch['modal_idx'], batch['token_height'], batch['in_pos'], batch['target_in_pos']  
    inf_out = model.inference(in_modal, in_pos, modal_idx, in_mask=in_mask, sampling_method="argmax", temperature=0.1, max_length=model.out_vocab.max_seq_len['lmx'])
    total_inf_out.append(inf_out.cpu())
    total_target_out.append(target_out.cpu())

from umust.evaluation_utils import calc_ser_metric

gold_lmx, pred_lmx = [], []
for inf_out, target_out in zip(total_inf_out, total_target_out):
  for idx in range(inf_out.shape[0]):

    gold = target_out[idx, :, 0] - model.out_vocab.idx_shifts['lmx']
    pred = inf_out[idx, :, 0] - model.out_vocab.idx_shifts['lmx']
    
    gold[gold<0] = 0
    pred[pred<0] = 0

    gold = out_idx_handler.vocabs['lmx'].decode(gold.cpu())
    pred = out_idx_handler.vocabs['lmx'].decode(pred.cpu())

    gold_lmx.append(gold)
    pred_lmx.append(pred)
print("\n\n")
print(run_path.name, iteration)
result = calc_ser_metric(gold_lmx, pred_lmx)
print(f"scanned: {result}")

olimpic_loader = dataset.get_specific_testset_loader('olimpic-synthetic', 'pt', 'lmx', batch_size=32, use_valid=True)

total_inf_out = []
total_target_out = []
i = 0
with torch.inference_mode():
  for batch in tqdm(olimpic_loader):
    in_modal, in_mask, target_in, target_out, modal_idx, token_heights, in_pos, target_in_pos = batch['in_modal'], batch['in_mask'], batch['target_in'], batch['target_out'], batch['modal_idx'], batch['token_height'], batch['in_pos'], batch['target_in_pos']  
    inf_out = model.inference(in_modal, in_pos, modal_idx, in_mask=in_mask, sampling_method="argmax", temperature=0.1, max_length=model.out_vocab.max_seq_len['lmx'])
    total_inf_out.append(inf_out.cpu())
    total_target_out.append(target_out.cpu())
gold_lmx, pred_lmx = [], []
for inf_out, target_out in zip(total_inf_out, total_target_out):
  for idx in range(inf_out.shape[0]):

    gold = target_out[idx, :, 0] - model.out_vocab.idx_shifts['lmx']
    pred = inf_out[idx, :, 0] - model.out_vocab.idx_shifts['lmx']
    
    gold[gold<0] = 0
    pred[pred<0] = 0

    gold = out_idx_handler.vocabs['lmx'].decode(gold.cpu())
    pred = out_idx_handler.vocabs['lmx'].decode(pred.cpu())

    gold_lmx.append(gold)
    pred_lmx.append(pred)

result = calc_ser_metric(gold_lmx, pred_lmx)
print(f"synthetic: {result}")