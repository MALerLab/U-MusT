#!/bin/bash

# this script is for Ubuntu 22.04
# it will install MuseScore 3.6.2

sudo apt update
sudo apt install -y \
  libnss3-dev libegl1-mesa-dev libglu1-mesa-dev \
  freeglut3-dev mesa-common-dev libjack-jackd2-dev \
  libxss1 libgconf-2-4 libxtst6 libxrandr2 \
  libasound2-dev libxss1 libgconf-2-4 \
  xvfb
wget -O mscore.AppImage https://github.com/musescore/MuseScore/releases/download/v3.6.2/MuseScore-3.6.2.548021370-x86_64.AppImage
chmod +x mscore.AppImage
./mscore.AppImage --appimage-extract
rm -r mscore.AppImage
mv squashfs-root mscore-3.6.2
xvfb-run -a ./mscore-3.6.2/AppRun -v