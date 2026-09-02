#!/usr/bin/env python3
"""
Image-to-Audio inference script based on custom_inference_i2a_musicxml.ipynb
Provides high-level functions for MusicXML to audio conversion pipeline
with continuous sheet music generation.
"""

import functools
import math
import os
import tempfile
from pathlib import Path
from typing import Iterator, Tuple, Union, List
import cv2
import numpy as np
import requests

DEVICE = "cuda"
import torch
import torchvision
import PIL.Image
from ultralytics import YOLO
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import shutil

# Import project modules
from umust.utils import *
from umust.data_decode_utils import TensorDecoder
from umust.evaluation_utils import LayerPeeper, use_attn_weights
from mxl_render_scripts.mxl_utils import convert_mxl_to_pdf, split_pdf


# Set Qt backend for headless operation
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def wandb_style_config_to_omega_config(wandb_conf):
    """Convert wandb config to omega config format"""
    for wandb_key in ["wandb_version", "_wandb"]:
        if wandb_key in wandb_conf:
            del wandb_conf[wandb_key]
    for key in wandb_conf:
        if 'desc' in wandb_conf[key]:
            del wandb_conf[key]['desc']
        if 'value' in wandb_conf[key]:
            wandb_conf[key] = wandb_conf[key]['value']
    return wandb_conf


YOLO_MODELS_URLS = {
    'ls-yolo-system-v2.0.0.pt': 'https://github.com/MALerLab/ls-yolo/releases/download/system-v2/ls-yolo-system-v2.0.0.pt',
    'ls-yolo-staff-height-v2.0.0.pt': 'https://github.com/MALerLab/ls-yolo/releases/download/staff-height-v2/ls-yolo-staff-height-v2.0.0.pt',
}


def load_yolo_model(checkpoint_name: str, checkpoint_dir: Union[str, Path] = 'yolo') -> YOLO:
    """Load a fine-tuned ls-yolo checkpoint, downloading it from the
    MALerLab/ls-yolo GitHub release if it is not present locally."""
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    if not checkpoint_path.exists():
        url = YOLO_MODELS_URLS[checkpoint_name]
        print(f"Downloading YOLO checkpoint {checkpoint_name} from {url} ...")
        r = requests.get(url, allow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download YOLO checkpoint {checkpoint_name} from {url}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(r.content)
    return YOLO(checkpoint_path)


def detect_systems_in_image(image: np.ndarray, yolo_system_model) -> List[Tuple[int, int, int, int, float]]:
    """Detect musical systems in an image using YOLO"""
    results = yolo_system_model([image])
    systems = []
    for result in results:
        bboxs = result.boxes.xyxy
        if len(bboxs) < 1:
            continue
        bboxs = bboxs.int().tolist()
        confs = result.boxes.conf.tolist()
        bboxs_with_conf = [(*coords, conf) for coords, conf in zip(bboxs, confs)]
        bboxs_with_conf = sorted(bboxs_with_conf, key=lambda x: (x[1], x[0]))
        systems.extend(bboxs_with_conf)
    return systems


def mxl_to_system_images(mxl_path: Union[str, Path], mscore_path: str, work_dir: Path) -> List[Path]:
    """Convert MusicXML to a list of system image file paths."""
    mxl_path = Path(mxl_path)
    score_name = mxl_path.stem
    
    pdf_dir = work_dir / 'pdf'
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{score_name}.pdf"

    print(f"Rendering MusicXML to PDF: {mxl_path} -> {pdf_path}")
    convert_mxl_to_pdf(mxl_path=mxl_path, out_path=pdf_path, script_path=mscore_path)

    img_dir = work_dir / 'image_pages'
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Splitting PDF into images: {pdf_path} -> {img_dir}")
    split_pdf(pdf_path=pdf_path, out_dir=img_dir, score_name=score_name)

    image_paths = sorted(list(img_dir.glob('*.png')))
    print(f"Found {len(image_paths)} page images.")

    yolo_system = load_yolo_model('ls-yolo-system-v2.0.0.pt')
    
    system_image_paths = []
    system_img_dir = work_dir / "system_images"
    system_img_dir.mkdir(parents=True, exist_ok=True)

    try:
        for page_idx, image_path in enumerate(image_paths):
            page_image = cv2.imread(str(image_path))
            page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
            systems = detect_systems_in_image(page_image, yolo_system)
            
            for system_idx, (lx, ly, rx, ry, conf) in enumerate(systems):
                if conf < 0.4:
                    continue
                system_image = page_image[ly:ry, lx:rx]
                sys_img_path = system_img_dir / f"page_{page_idx:02d}_system_{system_idx:02d}.png"
                
                # Convert to BGR for saving with OpenCV
                cv2.imwrite(str(sys_img_path), cv2.cvtColor(system_image, cv2.COLOR_RGB2BGR))
                system_image_paths.append(sys_img_path)
    finally:
        del yolo_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return system_image_paths


@functools.cache
def load_tokenizer(tokenizer_path: Union[str, Path]):
    tokenizer_path = Path(tokenizer_path)
    config_path = tokenizer_path / "files" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    wandb_config = OmegaConf.load(config_path)
    try:
        config = wandb_style_config_to_omega_config(wandb_config)
    except Exception:
        config = wandb_config

    vq_version = config.data.get('vq_model', None)
    if not vq_version:
        raise ValueError(
            f"Could not determine the image tokenizer from {config_path} (missing "
            f"data.vq_model). Refusing to guess a default, since the wrong tokenizer "
            f"silently produces mis-sized preprocessing (see README's Tokenizers section)."
        )

    config.data.dac_model = 'unidac4'
    config.data.data_dir = 'dummy'
    
    vq_model, _ = get_vq_model(config)
    return vq_model.eval(), config


@functools.cache
def load_llm(llm_path: Union[str, Path]):
    llm_path = Path(llm_path)
    _, config = load_tokenizer(llm_path)
    vq_model, vq_emb = get_vq_model(config)
    dac_model, dac_emb = get_dac_model(config)
    dataset = get_dataset(config)
    model = get_model(config, vq_emb, dac_emb, dataset)
    
    ckpt_paths = list((llm_path / 'files' / "checkpoints").glob("*.pt"))
    ckpt_paths = [x for x in ckpt_paths if x.name != 'last_checkpoint.pt']
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint found in {llm_path}/files/checkpoints/")
    
    last_ckpt_path = max(ckpt_paths, key=lambda p: int(p.stem.split('_')[0][4:]))
    
    state_dict = torch.load(last_ckpt_path, map_location="cpu")["model_state_dict"]
    load_model_state_dict(model, state_dict)
    
    model.eval()
    decoder = TensorDecoder(config, model.in_vocab, model.out_vocab, Path(tempfile.mkdtemp()), device=DEVICE)
    return model, decoder, config, dataset


def detect_staff_height(image: np.ndarray) -> float:
    yolo_staff = load_yolo_model('ls-yolo-staff-height-v2.0.0.pt')
    try:
        left_half = image[:, :image.shape[1]//2]
        if len(left_half.shape) == 2 or left_half.shape[2] == 1:
            left_half = cv2.cvtColor(left_half, cv2.COLOR_GRAY2RGB)
        
        results = yolo_staff([left_half])
        staff_heights = []
        for result in results:
            bboxs = result.boxes.xyxy.int().tolist()
            confs = result.boxes.conf.tolist()
            for (lx, ly, rx, ry), conf in zip(bboxs, confs):
                if conf > 0.4:
                    staff_heights.append(ry - ly)
        return (sum(staff_heights) / len(staff_heights)) if staff_heights else 20.0
    finally:
        del yolo_staff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def preprocess_system_image(image: np.ndarray, vq_version: str = 'unirqvae') -> torch.Tensor:
    TGT_HEIGHT = 18 if vq_version == 'unirqvae3' else 20
    staff_height = detect_staff_height(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    resize_ratio = TGT_HEIGHT / staff_height
    r_h, r_w = int(image.shape[0] * resize_ratio), int(image.shape[1] * resize_ratio)
    resized_img = cv2.resize(image, (r_w, r_h), interpolation=cv2.INTER_AREA)
    
    pil_image = PIL.Image.fromarray(resized_img)
    img_array = np.array(pil_image)
    median = np.median(img_array)
    img_array[img_array > (median-20)] = 255
    pil_image = PIL.Image.fromarray(img_array)
    
    tf = torchvision.transforms.ToTensor()
    nm = torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    tensor_image = tf(pil_image)
    
    h_padding = (16 - tensor_image.shape[-2] % 16) % 16
    w_padding = (16 - tensor_image.shape[-1] % 16) % 16
    tensor_image = torch.nn.functional.pad(tensor_image, (4, 3 + w_padding, 2, 1 + h_padding), mode='constant', value=1.0)
    return nm(tensor_image)


def tokenize_image(system_image: np.ndarray, model_path: str) -> torch.Tensor:
    vq_model, config = load_tokenizer(model_path)
    tensor_image = preprocess_system_image(system_image, config.data.vq_model)
    vq_model = vq_model.to(DEVICE)
    with torch.no_grad():
        tokens = vq_model.get_codes(tensor_image.to(DEVICE).unsqueeze(0))
    return tokens


def find_audio_border(attn: torch.Tensor, sep_idx: int, thr: float = 0.5, min_run: int = 3) -> int:
    """Finds the first audio token index that predominantly attends to image tokens after <SEP>."""
    head_avg = attn.mean(dim=1)
    post_sep_share = head_avg[:, sep_idx + 1:].sum(dim=1)
    over = (post_sep_share >= thr).float()
    run = torch.nn.functional.conv1d(over[None, None, :], weight=torch.ones(1, 1, min_run), padding=min_run - 1)[0, 0]
    idx = (run >= min_run).nonzero(as_tuple=True)[0]
    return idx[0].item() if idx.numel() else -1


def decode_audio_tokens(audio_tokens: torch.Tensor, model_path: str, audio_path: Path):
    model, decoder, config, dataset = load_llm(model_path)
    in_idx_handler = dataset.in_idx_handler
    pt_in_modal_idx = in_idx_handler.vocab_keys.index('pt')
    dac_out_modal_idx = dataset.out_idx_handler.vocab_keys.index('dac')
    modal_idx = torch.tensor([pt_in_modal_idx, dac_out_modal_idx])
    modal_idxs = modal_idx.unsqueeze(0).to(DEVICE).long()
    
    # We need a dummy token_heights, the actual one isn't used for dac-only decoding
    token_heights = torch.tensor([[0, 0]], dtype=torch.int16).to(DEVICE)

    # Create temp dir inside the output folder to avoid cross-device link errors
    with tempfile.TemporaryDirectory(dir=audio_path.parent) as temp_dir:
        temp_dir = Path(temp_dir)
        decoder(
            audio_tokens.to(DEVICE),
            modal_idxs[:, 1],
            'dummy',
            'prediction',
            token_heights=token_heights,
            use_in=False,
            custom_output_fns=[temp_dir / "output.wav"]
        )
        output_audio_files = list(temp_dir.glob('*.wav')) + list(temp_dir.glob('*:*.wav')) + list(temp_dir.glob('*.mp3'))
        if output_audio_files:
            generated_file = output_audio_files[0]
            if generated_file.suffix == '.mp3':
                from pydub import AudioSegment
                AudioSegment.from_mp3(generated_file).export(audio_path, format="wav")
            else:
                shutil.move(generated_file, audio_path)
        else:
            raise RuntimeError(f"Audio generation failed. No audio file in {temp_dir}")


def pt_collate_fn(pts, in_pos):
    max_len = max([x.shape[-2] for x in pts])
    in_modal = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[-2])) for x in pts]
    in_pos = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[-2])) for x in in_pos]

    in_modal = torch.cat(in_modal, dim=0).squeeze(1)
    in_pos = torch.stack(in_pos)

    in_mask = torch.ones((len(in_modal), max_len), dtype=torch.bool)
    in_mask[(in_modal[:, :, 0] == 0)] = 0
    return in_modal, in_pos, in_mask

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Convert MusicXML to continuous audio.")
    parser.add_argument("mxl_path", help="Path to the MusicXML file.")
    parser.add_argument("--instrument", choices=["piano", "strings"], default="piano", help="Instrument type; selects a released checkpoint under --models_dir. Ignored if --run_path is given.")
    parser.add_argument("-o", "--output", default="output", help="Output directory.")
    parser.add_argument("--mscore-path", default=None, help="Path to MuseScore executable (default: auto-detect).")
    parser.add_argument("--device", default="cuda", help="Device to run inference on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for inference.")
    parser.add_argument("--models_dir", default="models", help="Directory containing the released model checkpoints (used with --instrument).")
    parser.add_argument("--run_path", default=None, help="Training run directory containing files/config.yaml and files/checkpoints/*.pt (e.g. a model you trained yourself). Takes precedence over --instrument/--models_dir.")
    args = parser.parse_args()
    DEVICE = args.device

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mscore_path is None:
        candidates = ["/app/mscore-3.6.2/AppRun", shutil.which("mscore"), shutil.which("musescore3"), shutil.which("musescore")]
        args.mscore_path = next((c for c in candidates if c and Path(c).exists()), None)
        if args.mscore_path is None:
            raise FileNotFoundError(
                "MuseScore executable not found. Install MuseScore 3.6.2 and pass its path "
                "with --mscore-path (in headless/server environments run it under xvfb)."
            )

    if args.run_path is not None:
        model_path = args.run_path
        if not (Path(model_path) / "files" / "config.yaml").exists():
            raise FileNotFoundError(
                f"--run_path {model_path} does not look like a training run directory "
                f"(expected {model_path}/files/config.yaml)."
            )
    else:
        instrument_to_model = {
            "piano": "run-20250225_062905-9n1554as",
            "strings": "run-20250130_150202-x9znhap2"
        }
        model_dir_name = instrument_to_model[args.instrument]
        model_path = f"{args.models_dir}/{model_dir_name}/{model_dir_name}"
        if not Path(model_path).exists():
            model_path = f"{args.models_dir}/{model_dir_name}"
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. No pretrained checkpoints are publicly "
                f"released. Either train your own model (see README's Training section) and "
                f"pass --run_path pointing at the resulting run directory, or place a "
                f"checkpoint directory under {args.models_dir}/."
            )

    work_dir = Path(tempfile.mkdtemp())
    score_name = Path(args.mxl_path).stem

    print("Step 1: Converting MusicXML to system images...")
    system_image_paths = mxl_to_system_images(args.mxl_path, args.mscore_path, work_dir)

    print(f"Step 2: Tokenizing {len(system_image_paths)} system images...")
    token_paths = []
    token_dir = work_dir / "tokens"
    token_dir.mkdir()
    for i, img_path in enumerate(tqdm(system_image_paths, desc="Tokenizing")):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tokens = tokenize_image(image, model_path)
        token_path = token_dir / f"tokens_{i:03d}.pt"
        torch.save(tokens.cpu(), token_path)
        token_paths.append(token_path)

    print("Step 3: Performing continuous inference...")
    model, decoder, config, dataset = load_llm(model_path)
    model = model.to(args.device)
    in_idx_handler = dataset.in_idx_handler
    sep_token_id = in_idx_handler.img_crop_cat_sep_idx
    pt_in_modal_idx = in_idx_handler.vocab_keys.index('pt')
    dac_out_modal_idx = dataset.out_idx_handler.vocab_keys.index('dac')
    modal_idx = torch.tensor([pt_in_modal_idx, dac_out_modal_idx])

    condition = None
    out_tokens = []
    
    sheet_paths_dict = {score_name: token_paths}

    if len(token_paths) == 1:
        pt = torch.load(token_paths[0]).to(torch.int16).cpu().unsqueeze(0)
        data, token_height, pos = in_idx_handler([pt], 'pt', add_height_token=False)

        in_modal_prep, in_pos_prep, in_mask_prep = pt_collate_fn([data], [pos])

        modal_idxs = modal_idx.repeat(len(in_modal_prep), 1)

        in_modal = in_modal_prep.to(args.device).long()
        in_pos = in_pos_prep.to(args.device).long()
        in_mask = in_mask_prep.to(args.device).bool()
        modal_idxs = modal_idxs.to(args.device).long()
        token_heights = torch.stack([torch.tensor([token_height]), torch.zeros(len([token_height]))], dim=-1).to(torch.int16).to(args.device)

        with torch.no_grad():
            inferenced_output = model.inference(
                in_modal=in_modal, in_pos=in_pos, modal_idx=modal_idxs, in_mask=in_mask,
                token_heights=token_heights, sampling_method="none", threshold=0.9,
                temperature=1.0, manual_seed=args.seed, max_length=model.out_vocab.max_seq_len['dac'],
                condition=condition
            )

        out_tokens.append(inferenced_output)
    else:
        for j in tqdm(range(len(token_paths) - 1), desc="Continuous Inference"):
            temp_paths_dict = {score_name: sheet_paths_dict[score_name][j:j+2]}
            
            for sheet_name, pt_paths_list in temp_paths_dict.items():
                pt_for_sheet = []
                for pt_path in pt_paths_list:
                    pt = torch.load(pt_path).to(torch.int16).cpu().unsqueeze(0)
                    pt_for_sheet.append(pt)
                data, token_height, pos = in_idx_handler(pt_for_sheet, 'pt', add_height_token=False)

            in_modal_prep, in_pos_prep, in_mask_prep = pt_collate_fn([data], [pos])
            
            modal_idxs = modal_idx.repeat(len(in_modal_prep), 1)

            in_modal = in_modal_prep.to(args.device).long()
            in_pos = in_pos_prep.to(args.device).long()
            in_mask = in_mask_prep.to(args.device).bool()
            modal_idxs = modal_idxs.to(args.device).long()
            token_heights = torch.stack([torch.tensor([token_height]), torch.zeros(len([token_height]))], dim=-1).to(torch.int16).to(args.device)

            peeper = LayerPeeper(model.decoder.net.decoder.transformer_decoder.layers[-2][-2].attend, hook_fn=use_attn_weights)
            
            with torch.no_grad():
                inferenced_output = model.inference(
                    in_modal=in_modal, in_pos=in_pos, modal_idx=modal_idxs, in_mask=in_mask,
                    token_heights=token_heights, sampling_method="none", threshold=0.9,
                    temperature=1.0, manual_seed=args.seed, max_length=model.out_vocab.max_seq_len['dac'],
                    condition=condition
                )
            
            peeper.remove()

            if j != 0:
                decoder_hook_output = peeper.output[1:]
                decoder_hook_output = [peeper.output[0][:,:,k:k+1] for k in range(peeper.output[0].size(2))] + decoder_hook_output
            else:
                decoder_hook_output = peeper.output
            
            attn_weights = torch.stack(decoder_hook_output)
            attn_weights = attn_weights.squeeze(-2).squeeze(1)

            sep_idx = (in_modal == sep_token_id).nonzero(as_tuple=True)[1][0].item()
            
            audio_border = find_audio_border(attn=torch.stack([attn_weights[:,0], attn_weights[:,1], attn_weights[:,7], attn_weights[:,10]], dim=1).cpu(), sep_idx=sep_idx, thr=0.5, min_run=3)
            if audio_border < 0:
                audio_border = inferenced_output.size(1) - 1
            
            audio_border = audio_border - 10
            cond_length = (inferenced_output.size(1) - audio_border) // 5

            if j == 0:
                out_tokens.append(inferenced_output[:, :audio_border])
            elif j == len(token_paths) - 2:
                out_tokens.append(inferenced_output[:, 1:audio_border])
                out_tokens.append(inferenced_output[:, audio_border:])
            else:
                out_tokens.append(inferenced_output[:, 1:audio_border])

            condition = inferenced_output[:, audio_border : audio_border + cond_length]

    final_tokens = torch.cat(out_tokens, dim=1)
    print(f"Total audio tokens generated: {final_tokens.shape}")

    print("Step 4: Decoding final audio...")
    final_audio_path = output_dir / "final_output.wav"
    decode_audio_tokens(final_tokens, model_path, final_audio_path)

    print(f"Cleaning up temporary directory: {work_dir}")
    shutil.rmtree(work_dir)

    print(f"Inference complete! Final audio saved to: {final_audio_path}")
