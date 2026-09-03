import torch.nn as nn
from collections import OrderedDict


def transfer_weight_from_whisper(model, whisper_weight:OrderedDict):
  num_encoder_layers = len(model.encoder.encoder.transformer.layers) // 2
  num_decoder_layers = len(model.decoder.net.decoder.transformer_decoder.layers) // 3
  
  encoder_layers = model.encoder.encoder.transformer.layers
  decoder_layers = model.decoder.net.decoder.transformer_decoder.layers

  for i in range(num_encoder_layers):
    encoder_layers[i*2][1].to_q = nn.Linear(whisper_weight[f'encoder.blocks.{i}.attn.query.weight'].shape[1], whisper_weight[f'encoder.blocks.{i}.attn.query.weight'].shape[0], bias=True)
    encoder_layers[i*2][1].to_q.weight.data = whisper_weight[f'encoder.blocks.{i}.attn.query.weight']
    encoder_layers[i*2][1].to_q.bias.data = whisper_weight[f'encoder.blocks.{i}.attn.query.bias']
    encoder_layers[i*2][1].to_k.weight.data = whisper_weight[f'encoder.blocks.{i}.attn.key.weight']
    encoder_layers[i*2][1].to_v = nn.Linear(whisper_weight[f'encoder.blocks.{i}.attn.value.weight'].shape[1], whisper_weight[f'encoder.blocks.{i}.attn.value.weight'].shape[0], bias=True)
    encoder_layers[i*2][1].to_v.weight.data = whisper_weight[f'encoder.blocks.{i}.attn.value.weight']
    encoder_layers[i*2][1].to_v.bias.data = whisper_weight[f'encoder.blocks.{i}.attn.value.bias']
    encoder_layers[i*2][1].to_out = nn.Linear(whisper_weight[f'encoder.blocks.{i}.attn.out.weight'].shape[1], whisper_weight[f'encoder.blocks.{i}.attn.out.weight'].shape[0], bias=True)
    encoder_layers[i*2][1].to_out.weight.data = whisper_weight[f'encoder.blocks.{i}.attn.out.weight']
    encoder_layers[i*2][1].to_out.bias.data = whisper_weight[f'encoder.blocks.{i}.attn.out.bias']
    encoder_layers[i*2+1][0][0].ln = nn.LayerNorm(whisper_weight[f'encoder.blocks.{i}.attn_ln.weight'].shape[0], elementwise_affine=True)
    encoder_layers[i*2+1][0][0].ln.bias.data = whisper_weight[f'encoder.blocks.{i}.attn_ln.bias']
    encoder_layers[i*2+1][0][0].ln.weight.data = whisper_weight[f'encoder.blocks.{i}.attn_ln.weight']
    encoder_layers[i*2+1][1].ff[0][0].weight.data = whisper_weight[f'encoder.blocks.{i}.mlp.0.weight']
    encoder_layers[i*2+1][1].ff[0][0].bias.data = whisper_weight[f'encoder.blocks.{i}.mlp.0.bias']
    encoder_layers[i*2+1][1].ff[2].weight.data = whisper_weight[f'encoder.blocks.{i}.mlp.2.weight']
    encoder_layers[i*2+1][1].ff[2].bias.data = whisper_weight[f'encoder.blocks.{i}.mlp.2.bias']
    if i < num_encoder_layers-1:
      encoder_layers[i*2+2][0][0].ln = nn.LayerNorm(whisper_weight[f'encoder.blocks.{i}.mlp_ln.weight'].shape[0], elementwise_affine=True)
      encoder_layers[i*2+2][0][0].ln.bias.data = whisper_weight[f'encoder.blocks.{i}.mlp_ln.bias']
      encoder_layers[i*2+2][0][0].ln.weight.data = whisper_weight[f'encoder.blocks.{i}.mlp_ln.weight']
    
  for i in range(num_decoder_layers):
    decoder_layers[i*3][1].to_q = nn.Linear(whisper_weight[f'decoder.blocks.{i}.attn.query.weight'].shape[1], whisper_weight[f'decoder.blocks.{i}.attn.query.weight'].shape[0], bias=True)
    decoder_layers[i*3][1].to_q.bias.data = whisper_weight[f'decoder.blocks.{i}.attn.query.bias']
    decoder_layers[i*3][1].to_q.weight.data = whisper_weight[f'decoder.blocks.{i}.attn.query.weight']
    decoder_layers[i*3][1].to_k.weight.data = whisper_weight[f'decoder.blocks.{i}.attn.key.weight']
    decoder_layers[i*3][1].to_v = nn.Linear(whisper_weight[f'decoder.blocks.{i}.attn.value.weight'].shape[1], whisper_weight[f'decoder.blocks.{i}.attn.value.weight'].shape[0], bias=True)
    decoder_layers[i*3][1].to_v.bias.data = whisper_weight[f'decoder.blocks.{i}.attn.value.bias']
    decoder_layers[i*3][1].to_v.weight.data = whisper_weight[f'decoder.blocks.{i}.attn.value.weight']
    decoder_layers[i*3][1].to_out = nn.Linear(whisper_weight[f'decoder.blocks.{i}.attn.out.weight'].shape[1], whisper_weight[f'decoder.blocks.{i}.attn.out.weight'].shape[0], bias=True)
    decoder_layers[i*3][1].to_out.bias.data = whisper_weight[f'decoder.blocks.{i}.attn.out.bias']
    decoder_layers[i*3][1].to_out.weight.data = whisper_weight[f'decoder.blocks.{i}.attn.out.weight']
    decoder_layers[i*3+1][0][0].ln = nn.LayerNorm(whisper_weight[f'decoder.blocks.{i}.attn_ln.weight'].shape[0], elementwise_affine=True)
    decoder_layers[i*3+1][0][0].ln.bias.data = whisper_weight[f'decoder.blocks.{i}.attn_ln.bias']
    decoder_layers[i*3+1][0][0].ln.weight.data = whisper_weight[f'decoder.blocks.{i}.attn_ln.weight']
    decoder_layers[i*3+1][1].to_q = nn.Linear(whisper_weight[f'decoder.blocks.{i}.cross_attn.query.weight'].shape[1], whisper_weight[f'decoder.blocks.{i}.mlp.0.weight'].shape[0], bias=True)
    decoder_layers[i*3+1][1].to_q.bias.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.query.bias']
    decoder_layers[i*3+1][1].to_q.weight.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.query.weight']
    decoder_layers[i*3+1][1].to_k.weight.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.key.weight']
    decoder_layers[i*3+1][1].to_v = nn.Linear(whisper_weight[f'decoder.blocks.{i}.cross_attn.value.weight'].shape[1], whisper_weight[f'decoder.blocks.{i}.cross_attn.value.weight'].shape[0], bias=True)
    decoder_layers[i*3+1][1].to_v.bias.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.value.bias']
    decoder_layers[i*3+1][1].to_v.weight.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.value.weight']
    decoder_layers[i*3+1][1].to_out = nn.Linear(whisper_weight[f'decoder.blocks.{i}.cross_attn.out.weight'].shape[1], whisper_weight[f'decoder.blocks.{i}.cross_attn.out.weight'].shape[0], bias=True)
    decoder_layers[i*3+1][1].to_out.bias.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.out.bias']
    decoder_layers[i*3+1][1].to_out.weight.data = whisper_weight[f'decoder.blocks.{i}.cross_attn.out.weight']
    decoder_layers[i*3+2][0][0].ln = nn.LayerNorm(whisper_weight[f'decoder.blocks.{i}.cross_attn_ln.weight'].shape[0], elementwise_affine=True)
    decoder_layers[i*3+2][0][0].ln.bias.data = whisper_weight[f'decoder.blocks.{i}.cross_attn_ln.bias']
    decoder_layers[i*3+2][0][0].ln.weight.data = whisper_weight[f'decoder.blocks.{i}.cross_attn_ln.weight']
    decoder_layers[i*3+2][1].ff[0][0].weight.data = whisper_weight[f'decoder.blocks.{i}.mlp.0.weight']
    decoder_layers[i*3+2][1].ff[0][0].bias.data = whisper_weight[f'decoder.blocks.{i}.mlp.0.bias']
    decoder_layers[i*3+2][1].ff[2].weight.data = whisper_weight[f'decoder.blocks.{i}.mlp.2.weight']
    decoder_layers[i*3+2][1].ff[2].bias.data = whisper_weight[f'decoder.blocks.{i}.mlp.2.bias']
    if i < num_decoder_layers-1:
      decoder_layers[i*3+3][0][0].ln = nn.LayerNorm(whisper_weight[f'decoder.blocks.{i}.mlp_ln.weight'].shape[0], elementwise_affine=True)
      decoder_layers[i*3+3][0][0].ln.bias.data = whisper_weight[f'decoder.blocks.{i}.mlp_ln.bias']
      decoder_layers[i*3+3][0][0].ln.weight.data = whisper_weight[f'decoder.blocks.{i}.mlp_ln.weight']

  return model