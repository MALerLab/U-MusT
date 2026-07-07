FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Create application directory
WORKDIR /app

# Install all system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Base dependencies
    python3 \
    python3-pip \
    wget \
    unzip \
    git \
    xvfb \
    # MuseScore dependencies
    libnss3-dev \
    libegl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libjack-jackd2-dev \
    libxss1 \
    libgconf-2-4 \
    libxtst6 \
    libxrandr2 \
    libasound2-dev \
    # X11 utilities for headless operation
    xvfb \
    xauth \
    x11-utils \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    fluidsynth \
    fluid-soundfont-gm \
    && rm -rf /var/lib/apt/lists/*

# Install MuseScore 3.6.2 using AppImage inside /app
RUN wget -O mscore.AppImage https://github.com/musescore/MuseScore/releases/download/v3.6.2/MuseScore-3.6.2.548021370-x86_64.AppImage && \
    chmod +x mscore.AppImage && \
    ./mscore.AppImage --appimage-extract && \
    rm mscore.AppImage && \
    mv squashfs-root mscore-3.6.2 && \
    xvfb-run -a env QT_QPA_PLATFORM=xcb ./mscore-3.6.2/AppRun -v

# Ensure the extracted AppImage is executable by all users
RUN chmod -R 755 /app/mscore-3.6.2

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create sub-directory structure.
# The tokenizer and translation-model checkpoints are not distributed with
# this image; bind-mount them at runtime, e.g.
#   docker run -v /path/to/vq_models:/app/vq_models \
#              -v /path/to/dac_models:/app/dac_models \
#              -v /path/to/models:/app/models ...
RUN mkdir -p models \
    yolo \
    custom_input \
    output \
    vq_models \
    dac_models

# Download fine-tuned YOLO detectors (system + staff height) from MALerLab/ls-yolo releases
RUN wget -O yolo/ls-yolo-system-v2.0.0.pt https://github.com/MALerLab/ls-yolo/releases/download/system-v2/ls-yolo-system-v2.0.0.pt && \
    wget -O yolo/ls-yolo-staff-height-v2.0.0.pt https://github.com/MALerLab/ls-yolo/releases/download/staff-height-v2/ls-yolo-staff-height-v2.0.0.pt

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Make sure Python can find the umust module
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set the default command for MusicXML processing
CMD ["python3", "infer.py", "input.mxl", "--instrument=piano", "-o", "output/", "--mscore-path=/app/mscore-3.6.2/AppRun"]
