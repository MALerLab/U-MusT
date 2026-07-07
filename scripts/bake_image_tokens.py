"""Bake shift-augmented RQ-VAE image tokens (paper Sec. III-A3, Appendix B).

For every source image this produces a token tensor with 32 spatial-shift
variants (8 horizontal x 4 vertical, 1 px steps). With --augment, five
additional randomly-degraded copies are baked per shift (rotation, erosion/
dilatation, local/global pixel negation, contrast, brightness), giving the
layout used for software-rendered datasets (GrandStaff, OLiMPiC synthetic):

  without --augment: (8, 4, H/16, W/16, n_codebook)   -> image_tokens/<model>/yolo_shifted/
  with    --augment: (6, 8, 4, H/16, W/16, n_codebook) -> image_tokens/<model>/yolo_shifted_augmented/
                      variant 0 is the clean image.

Images are padded with white to a canvas one full token larger than the
image on each axis (left 12 px, top 10 px, right/bottom to the next
multiple of 16 plus one stride); the shift variants are 1-px crops of that
canvas. This exactly reproduces the layout of the released training tokens.

Example (OLiMPiC synthetic, augmented):
  python3 scripts/bake_image_tokens.py dataset/olimpic_dataset_yolo/olimpic-1.0-synthetic \
    --pattern "*_yolo_resized.png" --augment

Example (OLiMPiC scanned, shifts only):
  python3 scripts/bake_image_tokens.py dataset/olimpic_dataset_yolo/olimpic-1.0-scanned \
    --pattern "*_yolo_resized.png"

Example (GrandStaff train split only, augmented):
  python3 scripts/bake_image_tokens.py dataset/olimpic_dataset_yolo/grandstaff-lmx \
    --pattern "*_yolo_resized.jpg" --augment --train_manifest dataset_pair_paths/grandstaff-lmx.json
"""
import argparse
import json
import sys
from pathlib import Path

import PIL.Image
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import image_aug_utils
from umust.utils import load_vq_model_mm


def get_random_augmentation():
  augmentations = []
  is_applied = image_aug_utils.is_applied

  # rotation, [-1, 1] degrees
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda(lambda x: image_aug_utils.random_rotate(x, 1.))
    )

  # erosion / dilatation
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda(lambda x: image_aug_utils.random_erosion_dilatation(x))
    )

  # neighborhood pixel negation (3*3 window)
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda(lambda x: image_aug_utils.random_local_pixel_negation(x, 0.2))
    )

  # global pixel negation
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda(lambda x: image_aug_utils.random_global_pixel_negation(x, 0.01))
    )

  # adjust contrast, by factor [-1, 1]
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda(lambda x: image_aug_utils.random_adjust_contrast(x, -1., 1.))
    )

  # adjust brightness, by delta [-0.5, 0.2]
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda(lambda x: image_aug_utils.random_adjust_brightness(x, -0.5, 0.2))
    )

  return transforms.Compose(augmentations)


def collect_images(image_dir: Path, pattern: str, train_manifest: Path, skip_distorted: bool):
  image_path_list = sorted(image_dir.rglob(pattern))
  image_path_list = [p for p in image_path_list if not p.name.startswith('.')]
  if skip_distorted:
    image_path_list = [p for p in image_path_list if 'distorted' not in p.name]
  if train_manifest is not None:
    with open(train_manifest) as f:
      dataset = json.load(f)
    train_dirs = set()
    for item in dataset['train']:
      pt = item['pt'] if isinstance(item['pt'], str) else item['pt'][0]
      train_dirs.add(pt.split('/image_tokens')[0])
    image_path_list = [
      p for p in image_path_list
      if str(p.parent.relative_to(image_dir)) in train_dirs
    ]
  return image_path_list


def main():
  parser = argparse.ArgumentParser(description='Bake shift-augmented RQ-VAE image tokens.')
  parser.add_argument('image_dir', type=Path, help='root directory searched recursively for source images')
  parser.add_argument('--pattern', default='*_yolo_resized.png', help='glob pattern for source images')
  parser.add_argument('--vq_model', default='unirqvae3', help='image tokenizer name')
  parser.add_argument('--vq_model_dir', default='vq_models', help='directory holding the tokenizer checkpoints')
  parser.add_argument('--image_compress_factor', type=int, default=16)
  parser.add_argument('--codebook_size', type=int, default=1024)
  parser.add_argument('--n_codebook', type=int, default=4)
  parser.add_argument('--augment', action='store_true', help='additionally bake 5 randomly degraded variants per shift')
  parser.add_argument('--out_dirname', default=None, help='token sub-directory name (default: yolo_shifted[_augmented])')
  parser.add_argument('--train_manifest', type=Path, default=None, help='only bake images of samples in this manifest\'s train split')
  parser.add_argument('--skip_distorted', action='store_true', help='skip images with "distorted" in the file name')
  parser.add_argument('--skip_existing', action='store_true')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--device', default='cuda')
  args = parser.parse_args()

  torch.manual_seed(args.seed)

  cfg = OmegaConf.create({'data': {
    'vq_model': args.vq_model,
    'vq_model_dir': args.vq_model_dir,
    'image_compress_factor': args.image_compress_factor,
    'codebook_size': args.codebook_size,
    'n_codebook': args.n_codebook,
  }})
  model = load_vq_model_mm(cfg).to(args.device).eval()
  torch.set_grad_enabled(False)

  model_string = f'{args.vq_model}_f{args.image_compress_factor}_c{args.codebook_size}_k{args.n_codebook}'
  out_dirname = args.out_dirname or ('yolo_shifted_augmented' if args.augment else 'yolo_shifted')

  totensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5], [0.5])

  image_path_list = collect_images(args.image_dir, args.pattern, args.train_manifest, args.skip_distorted)
  print(f'{len(image_path_list)} images to bake')

  for image_path in tqdm(image_path_list):
    save_path = (image_path.parent / 'image_tokens' / model_string / out_dirname / image_path.stem).with_suffix('.pt')
    if args.skip_existing and save_path.exists():
      continue
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = PIL.Image.open(image_path).convert('L')
    image = totensor(image)

    # White padding: one extra 16-px stride beyond the next multiple of 16 on
    # each axis. The base placement is (top 2+8, left 4+8); the extra 7 px
    # horizontally / 3 px vertically are consumed by the shift crops below.
    h_padding = (16 - image.shape[-2] % 16) % 16
    w_padding = (16 - image.shape[-1] % 16) % 16
    image = torch.nn.functional.pad(image, (4 + 8, 3 + w_padding + 8, 2 + 8, 1 + h_padding + 8), mode='constant', value=1.0)

    n_aug = 6 if args.augment else 1
    augmented = []
    for a in range(n_aug):
      x_y_shifted_tokens = []
      for j in range(4):
        y_shifted_img = image[:, j:image.shape[-2] - 3 + j]
        x_shifted_imgs = []
        for i in range(8):
          img = y_shifted_img[..., i:y_shifted_img.shape[-1] - 7 + i]
          if a > 0:
            img = get_random_augmentation()(img.squeeze(0)).unsqueeze(0)
          x_shifted_imgs.append(normalize(img))
        x_shifted_imgs = torch.stack(x_shifted_imgs)
        out = model.get_codes(x_shifted_imgs.to(args.device))
        x_y_shifted_tokens.append(out.squeeze(0))
      x_y_shifted_tokens = torch.stack(x_y_shifted_tokens).transpose(0, 1)  # (x_shift, y_shift, h, w, k)
      augmented.append(x_y_shifted_tokens)

    tokens = torch.stack(augmented) if args.augment else augmented[0]
    torch.save(tokens.to(torch.int16).cpu(), str(save_path))


if __name__ == '__main__':
  main()
