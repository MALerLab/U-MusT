#!/bin/bash
# U-MusT multimodal training launcher.
# Set data.data_dir to the root directory of your preprocessed datasets.

# Image-to-Audio (I2A) direction, piano: OMR + MIDI-to-audio + image-to-audio
# (released "piano" checkpoint; reproduces Table VII)
python3 train_multimodal.py \
  --config-name=config_mm \
  data=omr_piano_synth_long \
  data.data_dir=dataset/ \
  train_params.world_size=2

# Image-to-Audio (I2A) direction, multi-instrument (released "strings" checkpoint)
# python3 train_multimodal.py \
#   --config-name=config_mm \
#   data=omr_direction_all \
#   data.data_dir=dataset/ \
#   train_params.world_size=2

# Audio-to-Image (A2I) direction: AMT + LMX-to-image + audio-to-image
# python3 train_multimodal.py \
#   --config-name=config_mm \
#   data=multimodal_amt_direction \
#   data.data_dir=dataset/ \
#   train_params.world_size=2
