#!/usr/bin/env bash
# Build EdgeTAM Tracker.app using PyInstaller inside the video_annotator conda env.
# Usage: bash build_app.sh
# Output: dist/EdgeTAM Tracker.app  (zip it to share)

set -euo pipefail

CONDA_ENV="video_annotator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Ensuring PyInstaller is installed in '$CONDA_ENV'..."
conda run -n "$CONDA_ENV" pip install pyinstaller --quiet

echo "==> Building app from $SCRIPT_DIR..."
cd "$SCRIPT_DIR"

# KMP_DUPLICATE_LIB_OK avoids the OpenMP crash that happens when torch is
# imported inside a conda run subprocess on Mac.
KMP_DUPLICATE_LIB_OK=TRUE \
  conda run -n "$CONDA_ENV" \
  pyinstaller tracker.spec --noconfirm

echo ""
echo "==> Build complete."
echo "    App:  $SCRIPT_DIR/dist/EdgeTAM Tracker.app"
echo ""
echo "    To share, zip it:"
echo "      cd dist && zip -r 'EdgeTAM Tracker.zip' 'EdgeTAM Tracker.app'"
echo ""
echo "    To create a .dmg (requires create-dmg):"
echo "      brew install create-dmg"
echo "      create-dmg 'EdgeTAM Tracker.dmg' 'dist/EdgeTAM Tracker.app'"
