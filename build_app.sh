#!/usr/bin/env bash
# Build EdgeTAM Tracker.app using PyInstaller inside the video_annotator conda env.
#
# Strategy: PyInstaller's BUNDLE puts Python packages in Contents/Frameworks/
# alongside native dylibs, which breaks codesign on macOS 15+. Instead we use
# COLLECT only (flat directory) and manually wrap it into a .app so everything
# stays in Contents/MacOS/ — a layout codesign handles without complaints.
#
# Usage:  bash build_app.sh
# Output: dist/EdgeTAM Tracker.app   (and a .zip beside it)

set -euo pipefail

CONDA_ENV="video_annotator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="EdgeTAM Tracker"
BUNDLE_ID="com.ctag.edgetam-tracker"
VERSION="1.0.0"

cd "$SCRIPT_DIR"

# ── 1. Install PyInstaller if needed ────────────────────────────────────────
echo "==> Ensuring PyInstaller is installed in '$CONDA_ENV'..."
conda run -n "$CONDA_ENV" pip install pyinstaller --quiet

# ── 2. Run PyInstaller (COLLECT only — no BUNDLE) ───────────────────────────
echo "==> Building with PyInstaller..."
KMP_DUPLICATE_LIB_OK=TRUE \
  conda run -n "$CONDA_ENV" \
  pyinstaller tracker.spec --noconfirm

COLLECT_DIR="$SCRIPT_DIR/dist/$APP_NAME"
APP_DIR="$SCRIPT_DIR/dist/$APP_NAME.app"

# ── 3. Create the .app bundle structure manually ────────────────────────────
echo "==> Assembling .app bundle..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"

# Move the entire COLLECT output into Contents/MacOS/ (flat layout)
cp -r "$COLLECT_DIR/." "$APP_DIR/Contents/MacOS/"

# Write Info.plist
cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>     <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>     <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
    <key>CFBundleName</key>           <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>    <string>APPL</string>
    <key>CFBundleShortVersionString</key> <string>$VERSION</string>
    <key>NSAppleScriptEnabled</key>   <false/>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key>       <string>NSApplication</string>
</dict>
</plist>
PLIST

# ── 4. Ad-hoc sign all Mach-O binaries then the bundle ──────────────────────
echo "==> Signing binaries..."
find "$APP_DIR/Contents/MacOS" -type f | while read f; do
  if file -b "$f" 2>/dev/null | grep -q "Mach-O"; then
    codesign --force --sign - "$f" 2>/dev/null
  fi
done

echo "==> Signing app bundle..."
codesign --force --sign - "$APP_DIR"

# Strip quarantine so recipients can double-click without a security warning
echo "==> Removing quarantine flag..."
xattr -cr "$APP_DIR"

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
