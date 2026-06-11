#!/usr/bin/env bash
# Build EdgeTAM Tracker.app using PyInstaller inside the video_annotator conda env.
#
# Strategy: Use PyInstaller BUNDLE to create the .app with the correct macOS
# layout (bootloader in Contents/MacOS/, Python in Contents/Frameworks/).
# Then clean up problematic directories that block codesign and sign everything.
#
# Usage:  bash build_app.sh
# Output: dist/EdgeTAM Tracker.app   (and a .zip beside it)

set -euo pipefail

CONDA_ENV="video_annotator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="EdgeTAM Tracker"

cd "$SCRIPT_DIR"

# ── 1. Install PyInstaller if needed ────────────────────────────────────────
echo "==> Ensuring PyInstaller is installed in '$CONDA_ENV'..."
conda run -n "$CONDA_ENV" pip install pyinstaller --quiet

# ── 2. Run PyInstaller (creates .app via BUNDLE) ─────────────────────────────
echo "==> Building with PyInstaller..."
KMP_DUPLICATE_LIB_OK=TRUE \
  conda run -n "$CONDA_ENV" \
  pyinstaller tracker.spec --noconfirm

APP_DIR="$SCRIPT_DIR/dist/$APP_NAME.app"

# ── 3. Clean up directories that block codesign ──────────────────────────────
echo "==> Cleaning up bundle..."

# .dist-info directories are treated as invalid sub-bundles by codesign
find "$APP_DIR" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

# .dylibs hidden directories (package-local dependency copies) are all
# duplicated at the top-level Frameworks/ — remove to avoid codesign failures
find "$APP_DIR" -name ".dylibs" -type d -exec rm -rf {} + 2>/dev/null || true

# Training configs are not needed for inference
rm -rf "$APP_DIR/Contents/Frameworks/_internal/sam2/configs/sam2.1_training" 2>/dev/null || true
rm -rf "$APP_DIR/Contents/Resources/_internal/sam2/configs/sam2.1_training" 2>/dev/null || true

# python3.11/ directory must look like a signable bundle to codesign
for PYDIR in \
    "$APP_DIR/Contents/Frameworks/_internal/python3.11" \
    "$APP_DIR/Contents/Frameworks/python3.11" \
    "$APP_DIR/Contents/Resources/python3.11"; do
  if [ -d "$PYDIR" ] && [ ! -f "$PYDIR/Info.plist" ]; then
    cat > "$PYDIR/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key> <string>com.ctag.edgetam.python311</string>
    <key>CFBundlePackageType</key> <string>FMWK</string>
    <key>CFBundleVersion</key>    <string>3.11</string>
</dict>
</plist>
PLIST
  fi
done

# ── 4. Ad-hoc sign everything ────────────────────────────────────────────────
echo "==> Signing all Mach-O files..."
find "$APP_DIR/Contents" -type f | while read -r f; do
  if file -b "$f" 2>/dev/null | grep -q "Mach-O"; then
    codesign --force --sign - "$f" 2>/dev/null || true
  fi
done || true

# Sign any python3.11 sub-bundle(s) we created
for PYDIR in \
    "$APP_DIR/Contents/Frameworks/_internal/python3.11" \
    "$APP_DIR/Contents/Frameworks/python3.11" \
    "$APP_DIR/Contents/Resources/python3.11"; do
  if [ -f "$PYDIR/Info.plist" ]; then
    codesign --force --sign - "$PYDIR" 2>/dev/null || true
  fi
done

echo "==> Signing app bundle..."
codesign --force --deep --sign - "$APP_DIR" 2>&1 || {
  echo "    Note: bundle sign reported errors (may be non-fatal on macOS 26)"
}

# Strip quarantine so recipients can double-click without a security warning
echo "==> Removing quarantine flag..."
xattr -cr "$APP_DIR" 2>/dev/null || true

# ── 5. Zip for distribution ──────────────────────────────────────────────────
echo "==> Creating zip..."
cd "$SCRIPT_DIR/dist"
rm -f "$APP_NAME.zip"
zip -r --symlinks "$APP_NAME.zip" "$APP_NAME.app" -q
cd "$SCRIPT_DIR"

echo ""
echo "==> Done!"
echo "    App: dist/$APP_NAME.app"
echo "    Zip: dist/$APP_NAME.zip"
echo ""
echo "    Share the .zip — recipients unzip and double-click the .app."
echo "    On first open they may need to right-click → Open to bypass Gatekeeper."
