import random
from collections import defaultdict
from pathlib import Path
from typing import Union, List

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms as VT


is_applied = lambda p: bool(torch.bernoulli(torch.tensor([p])).item())
get_rand_int = lambda l, u: torch.randint(l, u, (1,)).item()
get_rand_range = lambda l, u: torch.FloatTensor(1).uniform_(l, u).item()


def horizontal_shift(img:torch.Tensor, max_shift:int, a:int, fill=1.0) -> torch.Tensor:
  img = VT.functional.pad(img, (max_shift, 0, 0, 0), fill=fill)
  img = img[..., a:]
  
  return img

def random_horizontal_shift(img:torch.Tensor, max_shift:int, fill=1.0) -> torch.Tensor:
  a = get_rand_int(0, 2*max_shift+1)
  img = horizontal_shift(img, max_shift, a, fill)
  
  return img


def vertical_shift(img:torch.Tensor, max_shift:int, a:int, fill=1.0) -> torch.Tensor:
  height = img.shape[1]
  img = VT.functional.pad(img, (0, max_shift, 0, max_shift), fill=fill) # (left, top, right, bottom)
  img = img[:, a:, :]
  
  # crop to original height
  img = img[:, :height, :]
  
  return img

def random_vertical_shift(img:torch.Tensor, max_shift:int, fill=1.0) -> torch.Tensor:
  a = get_rand_int(0, 2*max_shift+1)
  img = vertical_shift(img, max_shift, a, fill)
  
  return img


def rotate(img:torch.Tensor, a:float, fill=1.0) -> torch.Tensor:
  if len(img.shape) == 2:
    img = img.unsqueeze(0)
  img = VT.functional.rotate(img, a, interpolation=VT.InterpolationMode.BILINEAR, fill=fill)
  if len(img.shape) == 3 and img.shape[0] == 1:
    img = img.squeeze(0)
  return img

def random_rotate(img:torch.Tensor, max_angle:float, fill=1.0) -> torch.Tensor:
  angle = get_rand_range(-max_angle, max_angle)
  img = rotate(img, angle, fill)
  
  return img


def adjust_brightness(img:torch.Tensor, delta:float) -> torch.Tensor:
  img = torch.clamp(img + delta, 0., 1.)
  
  return img

def random_adjust_brightness(img:torch.Tensor, l:float, u:float) -> torch.Tensor:
  delta = get_rand_range(l, u)
  img = adjust_brightness(img, delta)
  
  return img


def adjust_contrast(img:torch.Tensor, factor:float) -> torch.Tensor:
  factor = 2 ** factor
  if len(img.shape) == 2:
    mean = torch.mean(img, dim=(0, 1), keepdim=True)
  else:
    mean = torch.mean(img, dim=(1, 2), keepdim=True)
  img = torch.clamp((img - mean) * factor + mean, 0., 1.)
  
  return img

def random_adjust_contrast(img:torch.Tensor, l:float, u:float) -> torch.Tensor:
  factor = get_rand_range(l, u)
  img = adjust_contrast(img, factor)
  
  return img


def mask_pixel_negation(img:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
  img = mask * img + (1 - mask) * (1 - img)
  
  return img


def random_global_pixel_negation(img:torch.Tensor, p:float) -> torch.Tensor:
  mask = (torch.rand_like(img) >= get_rand_range(0, p)).float()
  img = mask_pixel_negation(img, mask)
  
  return img

def random_local_pixel_negation(img:torch.Tensor, p:float) -> torch.Tensor:
  mask = F.avg_pool2d(
    img.unsqueeze(0), 
    kernel_size=3, 
    stride=1, 
    padding=1,
    count_include_pad=False
  ) # pixel average of 3*3 neighborhood
  mask = mask.squeeze(0)
  
  mask = (mask <= 0.1) | (mask >= 0.9) # uniformly white or uniformly black
  mask = mask.float()
  
  mask = mask + (1 - mask) * ( torch.rand_like(mask) >= get_rand_range(0, p) ).float()
  
  img = mask_pixel_negation(img, mask)
  
  return img


def random_erosion_dilatation(img:torch.Tensor) -> torch.Tensor:
  d = get_rand_range(-np.pi/2, np.pi/2) # random angle
  d = torch.tensor([d], dtype=torch.float)
  x, y = torch.cos(d), 0.5 * torch.sin(d) # axis factor x:1, y:0.5
  
  # translate the image by x, y
  moved = VT.functional.affine(
    img.unsqueeze(0),  # Add batch dimension
    angle=0,
    translate=[x, y], 
    scale=1.0,
    shear=0.0,
    interpolation=VT.functional.InterpolationMode.BILINEAR,
    fill=1.0
  ).squeeze(0)  # Remove batch dimension
  
  
  if is_applied(0.5): # erosion
    img = torch.maximum(img, moved)
  else: # dilatation
    img = torch.clamp(img + moved - 1, 0., 1.)
  
  return img