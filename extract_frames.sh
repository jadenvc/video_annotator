#!/bin/bash
# Quick script to extract frames from a video for use with run_tracker.py

if [ $# -lt 1 ]; then
    echo "Usage: ./extract_frames.sh <video_file> [output_dir]"
    echo ""
    echo "Example:"
    echo "  ./extract_frames.sh examples/01_dog.mp4"
    echo "  ./extract_frames.sh examples/01_dog.mp4 my_frames"
    echo ""
    exit 1
fi

VIDEO="$1"
if [ ! -f "$VIDEO" ]; then
    echo "Error: Video file not found: $VIDEO"
    exit 1
fi

# Default output directory based on video name
if [ $# -ge 2 ]; then
    OUTPUT_DIR="$2"
else
    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
    OUTPUT_DIR="${BASENAME}_frames"
fi

echo "Extracting frames from: $VIDEO"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract frames with high quality
ffmpeg -i "$VIDEO" -q:v 2 -start_number 0 "$OUTPUT_DIR/%05d.jpg" -hide_banner

FRAME_COUNT=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo ""
echo "✓ Extracted $FRAME_COUNT frames to $OUTPUT_DIR/"
echo ""
echo "Now run:"
echo "  python run_tracker.py $OUTPUT_DIR/ --frames-dir"
