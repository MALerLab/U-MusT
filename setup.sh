#!/bin/bash
# U-MusT environment setup (tested on Ubuntu 22.04).
# Installs everything that pip cannot: system packages for rendering and
# synthesis, MuseScore 3.6.2, the GM soundfont link midi2audio expects,
# and (optionally) the fine-tuned YOLO detectors.
#
# Usage: ./setup.sh
set -e

# --- Python dependencies -----------------------------------------------------
pip install -r requirements.txt

# --- System packages ---------------------------------------------------------
# ffmpeg           : mp3 export of decoded audio (pydub)
# fluidsynth + GM  : MIDI -> audio rendering of decoded transcriptions
# xvfb             : virtual display for MuseScore on headless servers
# lib*             : MuseScore 3.6.2 AppImage runtime dependencies
sudo apt-get update
sudo apt-get install -y \
  ffmpeg fluidsynth fluid-soundfont-gm xvfb \
  libnss3-dev libegl1-mesa-dev libglu1-mesa-dev \
  freeglut3-dev mesa-common-dev libjack-jackd2-dev \
  libxss1 libxtst6 libxrandr2 libasound2-dev

# midi2audio only looks for the soundfont at this path
mkdir -p ~/.fluidsynth
if [ ! -e ~/.fluidsynth/default_sound_font.sf2 ]; then
  ln -s /usr/share/sounds/sf2/FluidR3_GM.sf2 ~/.fluidsynth/default_sound_font.sf2
fi

# --- MuseScore 3.6.2 ---------------------------------------------------------
if [ ! -x mscore-3.6.2/AppRun ]; then
  wget -O mscore.AppImage https://github.com/musescore/MuseScore/releases/download/v3.6.2/MuseScore-3.6.2.548021370-x86_64.AppImage
  chmod +x mscore.AppImage
  ./mscore.AppImage --appimage-extract
  rm mscore.AppImage
  mv squashfs-root mscore-3.6.2
fi
xvfb-run -a ./mscore-3.6.2/AppRun -v

# --- Fine-tuned YOLO detectors (optional pre-download) ------------------------
# infer.py downloads these automatically on first run; pre-fetch them here so
# offline machines are covered.
mkdir -p yolo
if [ ! -f yolo/ls-yolo-system-v2.0.0.pt ]; then
  wget -O yolo/ls-yolo-system-v2.0.0.pt https://github.com/MALerLab/ls-yolo/releases/download/system-v2/ls-yolo-system-v2.0.0.pt
fi
if [ ! -f yolo/ls-yolo-staff-height-v2.0.0.pt ]; then
  wget -O yolo/ls-yolo-staff-height-v2.0.0.pt https://github.com/MALerLab/ls-yolo/releases/download/staff-height-v2/ls-yolo-staff-height-v2.0.0.pt
fi

echo "Setup complete. Use --mscore-path $(pwd)/mscore-3.6.2/AppRun for inference,"
echo "or run MuseScore under xvfb-run on headless servers."
