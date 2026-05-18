# Video Annotator — Installation Guide

This guide installs the `run_tracker.py` GUI added on top of EdgeTAM.

## Prerequisites

- Git
- Conda or another Python environment manager
- Python 3.11 recommended
- `ffmpeg` for converting videos to JPEG frame directories

## Install

Clone the repo and create the `video_annotator` environment:

```bash
git clone https://github.com/jadenvc/video_annotator.git
cd video_annotator

conda create -n video_annotator python=3.11 -y
conda activate video_annotator
```

Install PyTorch. For Apple Silicon Macs, this is usually enough:

```bash
pip install torch torchvision
```

For CUDA/Linux, use the command recommended by
https://pytorch.org/get-started/locally/.

Install the project and tracker dependencies:

```bash
pip install -e ".[tracker]"
```

Download the EdgeTAM checkpoint:

```bash
bash checkpoints/download_ckpts.sh
```

Install `ffmpeg` if needed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```
---

## Verify Installation

Run this quick check to verify all dependencies:

```bash
python -c "
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from sam2.build_sam import build_sam2_video_predictor
print('✓ All dependencies installed')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ MPS available: {torch.backends.mps.is_available()}')
"
```

Expected output:
```
✓ All dependencies installed
✓ PyTorch ...
✓ MPS available: True/False
```

---

## Running the Tracker

### Basic Usage

```bash
# Extract frames from a video
./extract_frames.sh my_clip.mp4

# Run the tracker on the created frame directory
python run_tracker.py my_clip_frames/ --frames-dir

# Set custom confidence threshold (default: 0.50)
python run_tracker.py my_clip_frames/ --frames-dir --confidence 0.3
```

### Command-Line Options

```
usage: run_tracker.py [-h] [--confidence CONFIDENCE]
                      [--checkpoint CHECKPOINT] [--config CONFIG]
                      [--device DEVICE] [video]

positional arguments:
  video                 Path to an MP4 video file OR JPEG frame directory

options:
  -h, --help            show this help message and exit
  --confidence CONFIDENCE
                        Default confidence threshold (0.01-0.99)
  --checkpoint CHECKPOINT
                        Path to edgetam.pt checkpoint
  --config CONFIG       Hydra config name (default: edgetam.yaml)
  --device DEVICE       Force device (mps / cuda / cpu)
  --frames-dir          Input is a directory of JPEG frames
```

### Examples

```bash
# Use frame-directory input
python run_tracker.py shark_frames/ --frames-dir

# Lower confidence = track longer (may lose precision)
python run_tracker.py shark_frames/ --frames-dir --confidence 0.2

# Higher confidence = stop earlier (more precise)
python run_tracker.py shark_frames/ --frames-dir --confidence 0.7

# Force CPU (slower but more compatible)
python run_tracker.py shark_frames/ --frames-dir --device cpu
```

---

## How to Use the GUI

### 1. **Enable Tracker Mode**
   - Check the **"Tracker Mode"** checkbox in the right panel

### 2. **Choose Init Mode**
   - **Point**: Click once on a feature to track
   - **BBox**: Click and drag to draw a bounding box around a feature

### 3. **Configure Settings**
   - **Name**: Give your feature a descriptive name (e.g., "fish", "hand", "ball")
   - **Confidence**: Adjust the threshold (0.01-0.99)
     - **Higher** (0.7-0.9): Stops tracking when segmentation becomes uncertain (more conservative)
     - **Lower** (0.2-0.4): Keeps tracking even with lower confidence (more aggressive)

### 4. **Track a Feature**
   - Navigate to the frame where the feature first appears
   - With Tracker Mode ON, click (Point mode) or drag (BBox mode) on the feature
   - The tracker will automatically segment and propagate forward until:
     - Confidence drops below threshold, OR
     - End of video is reached

### 5. **Review & Manage**
   - Tracked features appear in the **"Tracked Features"** list
   - Click a feature to select it
   - **Delete**: Remove a tracked feature
   - **Rename**: Change the feature name (type new name, then click Rename)

### 6. **Export Results**
   - **Save JSON**: Export tracking data (masks, bboxes, metadata)
   - **Export Video**: Render video with overlaid segmentation masks

### 7. **Playback Controls**
   - **Play/Pause**: Play the video
   - **< Frame / Frame >**: Step backward/forward one frame
   - **Slider**: Scrub through the video

---

## Understanding Confidence Threshold

The **confidence threshold** determines when to stop tracking:

- **Model Output**: EdgeTAM produces segmentation masks with confidence scores (0-1)
- **Tracking Behavior**:
  - If confidence ≥ threshold → keep tracking
  - If confidence < threshold → **stop tracking**

### Recommended Settings

| Scenario | Threshold | Description |
|----------|-----------|-------------|
| **Clean, high-contrast features** | 0.6-0.8 | Stop when segmentation becomes slightly uncertain |
| **Balanced tracking** | 0.4-0.6 | Good default for most videos |
| **Challenging scenes** (occlusion, motion blur) | 0.2-0.4 | Keep tracking even with lower confidence |
| **Aggressive tracking** | 0.1-0.2 | Track as long as possible (may include errors) |

### Tips

- **Start higher** (0.6-0.7) to avoid false positives
- If tracking stops too early, **lower the threshold** and re-track
- If segmentation becomes inaccurate, **raise the threshold**

---

## Features

✨ **What the Tracker Does**

1. **Click or Draw Bounding Box** → Initialize tracking on any frame
2. **Automatic Segmentation** → EdgeTAM segments the feature using its lightweight RepViT backbone
3. **Forward Propagation** → Tracks the feature through subsequent frames
4. **Confidence-Based Stopping** → Automatically stops when confidence drops
5. **Multi-Feature Tracking** → Track multiple objects independently
6. **Color-Coded Visualization** → Each feature gets a unique color
7. **Export Options** → Save tracking data (JSON) or rendered video (MP4)

✨ **What's Different from run_api.py**

| Feature | run_api.py | run_tracker.py |
|---------|------------|----------------|
| **Purpose** | Manual bbox annotation | Automatic feature tracking |
| **Interaction** | Draw bbox for each frame | Click once, auto-propagate |
| **Segmentation** | Bounding boxes only | Pixel-accurate masks (EdgeTAM) |
| **Tracking** | Manual frame-by-frame | Automatic until confidence drops |
| **Model** | None (manual only) | EdgeTAM (SAM 2 variant) |
| **Init Mode** | BBox only | Point OR BBox |

---

## Performance on Apple Silicon

- **Device**: MPS (Metal Performance Shaders) — Apple's GPU acceleration
- **Expected FPS**: 5-15 FPS tracking speed (depends on video resolution)
- **Memory**: ~2-4 GB VRAM for 720p video
- **Model Size**: 56 MB (edgetam.pt checkpoint)

### Optimizations

- EdgeTAM uses a lightweight **RepViT backbone** optimized for edge devices
- Video frames are **offloaded to CPU** by default to save GPU memory
- Model runs in **eval mode** with no gradient computation

---

## Troubleshooting

### Issue: "Checkpoint not found"

```
Checkpoint not found: /path/to/video_annotator/checkpoints/edgetam.pt
Download with:  cd checkpoints && bash download_ckpts.sh
```

**Solution**: Download the checkpoint and make sure you are running from the repo root:

```bash
bash checkpoints/download_ckpts.sh
ls -lh checkpoints/edgetam.pt  # Should show ~54M file
```

### Issue: "ModuleNotFoundError: No module named 'decord'"

**Solution**:
```bash
pip install eva-decord
```

### Issue: "MPS backend not available"

**Solution**: MPS requires macOS 12.3+ and a recent PyTorch. If it is unavailable:

```bash
# Force CPU mode
python run_tracker.py my_clip_frames/ --frames-dir --device cpu
```

### Issue: Tracking stops immediately

**Possible causes**:
1. **Confidence threshold too high** → Lower it to 0.2-0.4
2. **Feature is occluded/moving too fast** → Try re-initializing on a clearer frame
3. **Initial click missed the feature** → Click more precisely on the feature center

### Issue: Tracking continues with wrong segmentation

**Possible causes**:
1. **Confidence threshold too low** → Raise it to 0.6-0.8
2. **Feature leaves the frame** → Model may latch onto background

### Issue: GUI is slow or unresponsive

**Solution**: Tracking runs in a background thread, but display can still lag on large videos. Try:
- Reduce video resolution before tracking
- Close other applications
- Use `--device cpu` if MPS has issues

---

## Example Workflow

1. **Launch the tracker**:
   ```bash
   ./extract_frames.sh examples/01_dog.mp4
   python run_tracker.py 01_dog_frames/ --frames-dir
   ```

2. **Enable Tracker Mode** (check the checkbox)

3. **Set confidence to 0.5** (default)

4. **Name the feature**: Type "dog" in the Name field

5. **Choose Point mode** (or BBox if you prefer)

6. **Click on the dog** in frame 0

7. **Wait for tracking** to complete (progress bar shows status)

8. **Review the result**:
   - Green overlay shows segmentation
   - Feature appears in the list: "dog (f0–f95)"

9. **Track another feature** (optional):
   - Seek to a different frame
   - Name it "ball"
   - Click on the ball
   - Now you have 2 tracked features

10. **Export**:
    - Click **"Export Video"** → Saves video with masks
    - Click **"Save JSON"** → Saves tracking metadata

---

## Advanced: Understanding the Code

### Key Classes

- **`TrackedFeature`**: Data class storing feature metadata (name, init frame, end frame, confidence threshold)
- **`FeatureStore`**: Manages all tracked features + per-frame masks/bboxes
- **`TrackingWorker`**: QThread that runs EdgeTAM propagation in the background (non-blocking UI)
- **`VideoLabel`**: Custom QLabel supporting both point clicks and bbox drag interactions
- **`MainWindow`**: Main GUI window with playback controls, feature list, and export

### Tracking Pipeline

1. **User clicks/drags** → `_on_point()` or `_on_bbox()`
2. **Map coordinates** → Label coords → Video frame coords
3. **Start worker thread** → `TrackingWorker.run()`
4. **EdgeTAM init** → `predictor.add_new_points_or_box()`
5. **Propagate forward** → `predictor.propagate_in_video()`
6. **For each frame**:
   - Get mask logits
   - Compute confidence = sigmoid(max_logit)
   - If confidence < threshold → **STOP**
   - Else → emit `frame_tracked` signal
7. **Worker done** → `_on_tracking_done()` → Add feature to store

### Confidence Calculation

```python
# In TrackingWorker.run()
logits = video_res_masks[0]  # Raw model output
mask = (logits > 0.0).cpu().numpy().squeeze()  # Binary mask

max_logit = logits.max().item()
conf = 1.0 / (1.0 + np.exp(-max_logit))  # Sigmoid activation
# conf ∈ [0, 1]

if conf < self.confidence_threshold:
    break  # Stop tracking
```

---

## File Structure

```
EdgeTAM/
├── run_tracker.py              ← NEW: Interactive feature tracker
├── gradio_app.py               ← Original web demo
├── checkpoints/
│   └── edgetam.pt              ← Model checkpoint (54 MB)
├── sam2/
│   ├── build_sam.py            ← Model builder
│   ├── sam2_video_predictor.py ← Video tracking API
│   ├── configs/
│   │   └── edgetam.yaml        ← Model config
│   └── utils/
│       └── misc.py             ← Video loading utilities
└── examples/                   ← 24 example videos
    ├── 01_dog.mp4
    ├── 02_cups.mp4
    └── ...
```

---

## JSON Output Format

When you click **"Save JSON"**, the tracker exports this format:

```json
{
  "video_path": "/path/to/video.mp4",
  "fps": 30.0,
  "total_frames": 120,
  "video_w": 1280,
  "video_h": 720,
  "features": [
    {
      "name": "shark",
      "init_frame": 0,
      "end_frame": 95,
      "init_type": "point",
      "init_coords": [640, 360],
      "confidence_threshold": 0.5,
      "color_idx": 0,
      "bboxes": {
        "0": [580, 320, 700, 400],
        "1": [582, 322, 702, 402],
        ...
      }
    }
  ]
}
```

**Note**: Masks are NOT saved in JSON (too large). Only bboxes are saved. If you need masks, use the exported video or modify the code to save masks as PNG sequences.

---

## Next Steps

- **Try it out**: `./extract_frames.sh examples/01_dog.mp4 && python run_tracker.py 01_dog_frames/ --frames-dir`
- **Experiment with confidence**: Try values from 0.2 to 0.8
- **Track multiple features**: Click on different objects in different frames
- **Export and review**: See how the masks look in the final video

Enjoy tracking! 🎯
