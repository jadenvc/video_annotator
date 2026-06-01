# Builds CTAG Annotator.exe for Windows.
# Run this ON a Windows machine with conda installed.
# Usage: powershell -ExecutionPolicy Bypass -File build_annotator_windows.ps1 [conda-env]
param([string]$CondaEnv = "video_annotator")

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==> Installing PyInstaller..."
conda run -n $CondaEnv pip install pyinstaller --quiet

Write-Host "==> Building..."
# Use a Windows-specific spec (no BUNDLE — that's macOS only)
conda run -n $CondaEnv pyinstaller `
    --noconfirm `
    --windowed `
    --name "CTAG Annotator" `
    --hidden-import PyQt6 `
    --hidden-import cv2 `
    --hidden-import matplotlib `
    --hidden-import matplotlib.backends.backend_agg `
    --exclude-module torch `
    --exclude-module sam2 `
    run_annotator.py

Write-Host ""
Write-Host "Done! dist\CTAG Annotator\"
Write-Host "Zip the 'CTAG Annotator' folder and share it."
