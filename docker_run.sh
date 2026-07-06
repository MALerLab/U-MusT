#!/bin/bash

# Docker run script for MusicXML to audio inference
# Usage: ./docker_run.sh <input_mxl> --instrument=<piano|strings> <output_dir> [--gpu=<gpu_index>] [--mscore-path=<path_to_musescore>]

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 <input_mxl> --instrument=<piano|strings> <output_dir> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  input_mxl          Path to input MusicXML file (.mxl or .musicxml)"
    echo "  --instrument       Instrument type: 'piano' or 'strings'"
    echo "  output_dir         Path to output directory on host"
    echo ""
    echo "Options:"
    echo "  --gpu=<index>      (Optional) GPU index to use (default: all GPUs)"
    echo "  --mscore-path=<path> (Optional) Path to the MuseScore executable on the host."
    echo "                     Defaults to /usr/bin/musescore3 if not provided."
    echo ""
    echo "Example:"
    echo "  $0 /path/to/sheet.mxl --instrument=piano ./output"
    echo "  $0 /path/to/sheet.musicxml --instrument=strings ./output --gpu=0"
    echo "  $0 /path/to/sheet.mxl --instrument=piano ./output --mscore-path=/Applications/MuseScore\\ 3.app/Contents/MacOS/mscore"
}

# Check for minimum number of arguments
if [ "$#" -lt 3 ]; then
    show_usage
    exit 1
fi

# Parse arguments
INPUT_MXL="$1"
INSTRUMENT_ARG="$2"
OUTPUT_DIR="$3"
shift 3
OTHER_ARGS="$@"

# Set defaults
GPU_INDEX=""
MSCORE_PATH="/app/mscore-3.6.2/AppRun" # Default path inside the container
MSCORE_HOST_PATH=""

# Parse optional arguments
for arg in $OTHER_ARGS; do
    case $arg in
        --gpu=*)
        GPU_INDEX="${arg#*=}"
        shift
        ;;
        --mscore-path=*)
        MSCORE_HOST_PATH="${arg#*=}"
        shift
        ;;
        *)
        echo "Error: Unknown option $arg"
        show_usage
        exit 1
        ;;
    esac
done


# Parse instrument argument
if [[ "$INSTRUMENT_ARG" =~ ^--instrument=(.+)$ ]]; then
    INSTRUMENT="${BASH_REMATCH[1]}"
else
    echo "Error: Invalid instrument argument. Must be --instrument=piano or --instrument=strings"
    show_usage
    exit 1
fi

# Validate instrument value
if [ "$INSTRUMENT" != "piano" ] && [ "$INSTRUMENT" != "strings" ]; then
    echo "Error: Instrument must be 'piano' or 'strings', got: $INSTRUMENT"
    show_usage
    exit 1
fi

# Get absolute paths
INPUT_MXL=$(realpath "$INPUT_MXL")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Validate input MusicXML exists
if [ ! -f "$INPUT_MXL" ]; then
    echo "Error: Input MusicXML file not found: $INPUT_MXL"
    exit 1
fi

# Create output directory with proper permissions
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Get current user ID and group ID to maintain permissions
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(whoami)

# Create a temporary home directory for the container user to avoid permission errors
DOCKER_HOME_DIR=$(realpath ./.docker_home)
mkdir -p "$DOCKER_HOME_DIR"

# Docker image name
IMAGE_NAME="latent-score-amt"

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .
echo "Docker build complete!"
echo ""

# Prepare docker command
DOCKER_CMD="docker run --rm"

# Set GPU arguments
if [ -n "$GPU_INDEX" ]; then
    DOCKER_CMD="$DOCKER_CMD --gpus device=$GPU_INDEX"
    echo "Running inference with GPU: $GPU_INDEX"
else
    DOCKER_CMD="$DOCKER_CMD --gpus all"
    echo "Running inference with all available GPUs"
fi

# Add user home directory
DOCKER_CMD="$DOCKER_CMD -e MPLCONFIGDIR=/home/$USER_NAME/.config/matplotlib"
DOCKER_CMD="$DOCKER_CMD -v $DOCKER_HOME_DIR:/home/$USER_NAME"


# Mount volumes
DOCKER_CMD="$DOCKER_CMD -v /etc/passwd:/etc/passwd:ro"
DOCKER_CMD="$DOCKER_CMD -v /etc/group:/etc/group:ro"
DOCKER_CMD="$DOCKER_CMD -v $INPUT_MXL:/app/input.mxl:ro"
DOCKER_CMD="$DOCKER_CMD -v $OUTPUT_DIR:/app/output"

# Prepare inference command arguments
INFER_CMD_ARGS="input.mxl --instrument=$INSTRUMENT -o /app/output"

# Handle MuseScore path
if [ -n "$MSCORE_HOST_PATH" ]; then
    MSCORE_HOST_PATH=$(realpath "$MSCORE_HOST_PATH")
    if [ ! -f "$MSCORE_HOST_PATH" ]; then
        echo "Error: MuseScore executable not found at specified host path: $MSCORE_HOST_PATH"
        exit 1
    fi
    # Mount the host executable to the default path inside the container
    DOCKER_CMD="$DOCKER_CMD -v $MSCORE_HOST_PATH:$MSCORE_PATH:ro"
    echo "Using MuseScore from host path: $MSCORE_HOST_PATH"
else
    echo "Using default MuseScore path from Docker image: $MSCORE_PATH"
fi
INFER_CMD_ARGS="$INFER_CMD_ARGS --mscore-path=$MSCORE_PATH"

# Final command assembly
# We run a bash command that, as root, fixes the output directory's permissions,
# and then switches to the host user to run the actual inference script.
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME bash -c \"chown $USER_ID:$GROUP_ID /app/output && su -s /bin/bash $USER_NAME -c 'python3 infer.py $INFER_CMD_ARGS'\""

# Print details
echo "----------------------------------------"
echo "Input MusicXML: $INPUT_MXL"
echo "Instrument: $INSTRUMENT"
echo "Output directory: $OUTPUT_DIR"
echo "Docker image: $IMAGE_NAME"
echo "----------------------------------------"
echo "Executing command:"
echo "$DOCKER_CMD"
echo "----------------------------------------"

# Run the container
eval $DOCKER_CMD

echo ""
echo "Inference complete! Check output in: $OUTPUT_DIR"
