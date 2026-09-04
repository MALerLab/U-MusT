"""Gradio demo for the U-MusT Image-to-Audio (piano) model.

`demo.engine` holds the framework-agnostic inference code (model loading,
YOLO system cropping, RQ-VAE tokenization, OMR, MIDI-to-audio, image-to-audio
and the Contin-U two-system sliding window). `app.py` at the repository root
wires it into a four-tab Gradio interface.
"""
