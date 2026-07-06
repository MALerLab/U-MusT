#!/bin/bash
# U-MusT multimodal training launcher.
# Set data.data_dir to the root directory of your preprocessed datasets.

# Image-to-Audio (I2A) direction: OMR + MIDI-to-audio + image-to-audio
python3 train_multimodal.py \
  --config-name=config_mm \
  data=multimodal_omr_direction_yolo \
  data.data_dir=dataset/ \
  train_params.world_size=2

# Audio-to-Image (A2I) direction: AMT + LMX-to-image + audio-to-image
# python3 train_multimodal.py \
#   --config-name=config_mm \
#   data=multimodal_amt_direction \
#   data.data_dir=dataset/ \
#   train_params.world_size=2
