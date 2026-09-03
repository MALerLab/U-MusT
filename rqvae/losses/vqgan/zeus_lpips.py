import math

import torch
from torch import nn
import torch.nn.functional as F

from collections import namedtuple

class ZeusLPIPS(nn.Module):
  # Learned perceptual metric
  def __init__(self, use_dropout=True):
    super().__init__()
    self.scaling_layer = ScalingLayer()
    self.net = ZeusEyeballWrapper(pretrained_weights_path="zeus/zeus_eyeball_yolo_resized.pt")
    for param in self.parameters():
        param.requires_grad = False
    self.loss_weights = [1, 2, 5, 10, 10]

  def forward(self, input, target, reduction='mean'):
    in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
    outs0, outs1 = self.net(in0_input), self.net(in1_input)
    
    total_loss = 0
    for out0, out1, loss_weight in zip(outs0, outs1, self.loss_weights):
      # Apply normalize_tensor
      out0_norm = normalize_tensor(out0)
      out1_norm = normalize_tensor(out1)
      
      # Calculate MSE loss
      loss = torch.nn.MSELoss()(out0_norm, out1_norm)

      loss = loss * loss_weight * 2 # 2 to make the scale similar to LPIPS
      total_loss += loss
    
    if reduction == 'none':
      return total_loss
    elif reduction == 'mean':
      return total_loss.mean()
    elif reduction == 'sum':
      return total_loss.sum()


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return (inp + 1) * 0.5  # Convert from [-1,1] to [0,1] range


class ZeusEyeballWrapper(nn.Module):
  def __init__(self, requires_grad=False, pretrained_weights_path="zeus/zeus_eyeball_yolo_resized.pt"):
    super().__init__()

    loaded = torch.load(pretrained_weights_path)
    config = loaded['config']
    zeus_eyeball_pretrained_features = ZeusEyeball(
                                                  config['dim'],
                                                  config['cnn_ch'],
                                                  config['cnn_stages'],
                                                  config['cnn_resblocks'],
                                                  conv_layer_name=config['conv_layer_name'],
                                                )
    zeus_eyeball_pretrained_features.load_state_dict(loaded['model'])
    zeus_eyeball_pretrained_features = zeus_eyeball_pretrained_features.conv

    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    self.N_slices = 5
    for x in range(3):
        self.slice1.add_module(str(x), zeus_eyeball_pretrained_features[x])
    for x in range(3, 5):
        self.slice2.add_module(str(x), zeus_eyeball_pretrained_features[x])
    for x in range(5, 7):
        self.slice3.add_module(str(x), zeus_eyeball_pretrained_features[x])
    for x in range(7, 8):
        self.slice4.add_module(str(x), zeus_eyeball_pretrained_features[x])
    for x in range(8, 9):
        self.slice5.add_module(str(x), zeus_eyeball_pretrained_features[x])
    if not requires_grad:
        for param in self.parameters():
            param.requires_grad = False

  def forward(self, x):
    h = self.slice1(x)
    h_relu1_2 = h
    h = self.slice2(h)
    h_relu2_2 = h
    h = self.slice3(h)
    h_relu3_2 = h
    h = self.slice4(h)
    h_relu4_1 = h
    h = self.slice5(h)
    h_relu5_1 = h
    zeus_outputs = namedtuple("ZeusOutputs", ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_1', 'relu5_1'])
    out = zeus_outputs(h_relu1_2, h_relu2_2, h_relu3_2, h_relu4_1, h_relu5_1)
    return out

class TFConv2d(torch.nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
  

  def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
    """
    i: input size (height or width)
    k: kernel size
    s: stride
    d: dilation
    """
    o = math.ceil(i / s) # output size
    return max((o - 1) * s + (k - 1) * d + 1 - i, 0)
  
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]

    pad_h = self._calc_same_pad(
      i=h, 
      k=self.kernel_size[0], 
      s=self.stride[0], 
      d=self.dilation[0]
    )
    pad_w = self._calc_same_pad(
      i=w, 
      k=self.kernel_size[1], 
      s=self.stride[1], 
      d=self.dilation[1]
    )

    if pad_h > 0 or pad_w > 0:
      x = F.pad(
        x, 
        [
          pad_w // 2, # left
          pad_w - pad_w // 2, # right
          pad_h // 2, # top
          pad_h - pad_h // 2 # bottom
        ]
      )
    

    return super().forward(x)

# ResNet blocks
class ResNetLike(nn.Module):
  def __init__(self, ConvClass, in_channels, out_channels, downsample=False):
    super().__init__()
    
    self.residual_layer = nn.Identity()
    self.downsample = downsample
    
    if downsample:
      self.residual_layer = nn.Sequential(
        ConvClass(in_channels, out_channels, kernel_size=3, stride=2, bias=False),
        nn.BatchNorm2d(out_channels) 
      )
    
    conv_layers = []
    
    if downsample:
      conv_layers.append(
        ConvClass(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
      )
    else:
      conv_layers.append(
        ConvClass(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
      )
    
    conv_layers.extend([
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      ConvClass(out_channels, out_channels, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(out_channels)
    ])
    
    self.conv_stack = nn.Sequential( *conv_layers )
    
    self.relu = nn.ReLU(inplace=True)
  
  
  def forward(self, x):
    residual = self.residual_layer(x)
    hidden = self.conv_stack(x)
    
    hidden = hidden + residual
    hidden = self.relu(hidden)
    
    return hidden

# FOR PERCEPTUAL LOSS
class ZeusEyeball(nn.Module):
  def __init__(
    self, 
    dim,
    cnn_ch, 
    cnn_stages, 
    cnn_resblocks, 
    conv_layer_name='Conv2d',
  ):
    super().__init__()
    
    self.cnn_stages = cnn_stages
    
    ConvClass = conv_layer_dict[conv_layer_name]
    
    layers = []
    
    # Initial convolutional layer
    layers.append(
      ConvClass(1, cnn_ch, kernel_size=3, stride=1, padding=1, bias=False)
    )
    
    in_channels = cnn_ch
    
    # add residual blocks
    for i in range(cnn_stages):
      out_channels = min(dim, cnn_ch * (2 ** i))
      layers.append(ResNetLike(ConvClass, in_channels, out_channels, downsample=True)) # only first layer do downsample
      
      for _ in range(cnn_resblocks - 1):
        layers.append(ResNetLike(ConvClass, out_channels, out_channels, downsample=False))
      
      in_channels = out_channels
    
    self.conv = nn.Sequential(*layers)


  def forward(self, x):
    return self.conv(x)


conv_layer_dict = {
  'Conv2d': nn.Conv2d,
  'TFConv2d': TFConv2d,
}


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

def channel_spatial_average(x, keepdim=True):
    return x.mean([1,2,3],keepdim=keepdim)