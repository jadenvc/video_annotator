#!/usr/bin/env bash
# Builds CTAG Annotator.app — no torch/SAM2 required.
# Requires: conda env with PyQt6, opencv-python, numpy, matplotlib, pyinstaller
# Usage: bash build_annotator_mac.sh [conda-env-name]
set -euo pipefail
CONDA_ENV="${1:-video_annotator}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="CTAG Annotator"
cd "$SCRIPT_DIR"

echo "==> Installing PyInstaller..."
conda run -n "$CONDA_ENV" pip install pyinstaller --quiet

echo "==> Building..."
conda run -n "$CONDA_ENV" pyinstaller annotator.spec --noconfirm

APP_DIR="$SCRIPT_DIR/dist/$APP_NAME.app"

echo "==> Cleaning up..."
find "$APP_DIR" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true
find "$APP_DIR" -name ".dylibs" -type d -exec rm -rf {} + 2>/dev/null || true

# python3.11/ needs to look like a bundle for codesign
for PYDIR in "$APP_DIR/Contents/Frameworks/python3.11" "$APP_DIR/Contents/Resources/python3.11"; do
  if [ -d "$PYDIR" ] && [ ! -f "$PYDIR/Info.plist" ]; then
    cat > "$PYDIR/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleIdentifier</key><string>com.ctag.annotator.python311</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleVersion</key><string>3.11</string>
</dict></plist>
PLIST
  fi
done

echo "==> Signing..."
find "$APP_DIR/Contents" -type f | while read -r f; do
  file -b "$f" 2>/dev/null | grep -q "Mach-O" && codesign --force --sign - "$f" 2>/dev/null || true
done || true
for PYDIR in "$APP_DIR/Contents/Frameworks/python3.11" "$APP_DIR/Contents/Resources/python3.11"; do
  [ -f "$PYDIR/Info.plist" ] && codesign --force --sign - "$PYDIR" 2>/dev/null || true
done
codesign --force --deep --sign - "$APP_DIR" 2>&1 || echo "  (bundle sign warnings — may be non-fatal)"
xattr -cr "$APP_DIR" 2>/dev/null || true

echo "==> Zipping..."
cd "$SCRIPT_DIR/dist"
rm -f "$APP_NAME.zip"
zip -r --symlinks "$APP_NAME.zip" "$APP_NAME.app" -q
cd "$SCRIPT_DIR"

echo ""
echo "Done! dist/$APP_NAME.app  (zip: dist/$APP_NAME.zip)"
echo "Recipients: right-click -> Open on first launch to bypass Gatekeeper."
