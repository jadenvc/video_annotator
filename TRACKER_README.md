# EdgeTAM Feature Tracker

**Automatic video feature tracking with pixel-accurate segmentation masks**

Built on EdgeTAM (SAM 2 variant optimized for edge devices), this interactive GUI tool lets you click on any feature in a video and automatically track it through subsequent frames until the segmentation confidence drops below your threshold.

---

## ⚡ Quick Start (Python 3.11 — No Decord)

Since you're on Python 3.11 and `decord` requires Python 3.8-3.10, use **frame directories**:

```bash
cd EdgeTAM

# 1. Extract frames from video
./extract_frames.sh examples/01_dog.mp4

# 2. Run tracker
python run_tracker.py 01_dog_frames/ --frames-dir
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed frame extraction guide.

---

## 🎯 Features

- ✨ **Click-to-track** — Point click or bounding box to initialize
- 🎨 **Pixel-accurate masks** — EdgeTAM segments with SAM 2 quality
- 📹 **Auto-propagation** — Tracks forward until confidence threshold
- 🎚️ **Confidence control** — Adjustable 0.01-0.99 (when to stop)
- 🔢 **Multi-object** — Track multiple features independently
- 💾 **Export** — Save JSON metadata or render MP4 with overlays
- 🧵 **Non-blocking** — Tracking runs in background thread
- 🍎 **M2 optimized** — Uses MPS (Metal) GPU acceleration

---

## 📋 Requirements

Install these in a fresh environment:

```bash
conda create -n video_annotator python=3.11 -y
conda activate video_annotator
pip install torch torchvision
pip install -e ".[tracker]"
bash checkpoints/download_ckpts.sh
```

You also need `ffmpeg` to extract video frames:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

---

## 🚀 Usage

### Option 1: Frame Directory (Recommended for Python 3.11)

```bash
# Extract frames
./extract_frames.sh your_video.mp4

# Run tracker
python run_tracker.py your_video_frames/ --frames-dir
```

### Option 2: Direct Video (Requires Python 3.8-3.10 + decord)

```bash
# Install decord first (only works on Python 3.8-3.10)
pip install eva-decord

# Run with video file
python run_tracker.py your_video.mp4
```

### Command-Line Options

```
python run_tracker.py <path> [options]

Arguments:
  path                  Video file OR directory of JPEG frames

Options:
  --frames-dir          Input is a directory of frames (not video)
  --confidence FLOAT    Threshold to stop tracking (default: 0.50)
  --device DEVICE       Force device: mps/cuda/cpu (default: auto)
  --checkpoint PATH     Custom EdgeTAM checkpoint
  --config NAME         Hydra config (default: edgetam.yaml)
```

### Examples

```bash
# Basic frame directory usage
python run_tracker.py frames/ --frames-dir

# Low confidence = track longer
python run_tracker.py frames/ --frames-dir --confidence 0.3

# High confidence = stop earlier (more precise)
python run_tracker.py frames/ --frames-dir --confidence 0.7

# Force CPU if MPS has issues
python run_tracker.py frames/ --frames-dir --device cpu
```

---

## 🖱️ GUI Controls

### Tracking Workflow

1. **Enable Tracker Mode** (checkbox in right panel)
2. **Choose init mode:**
   - **Point**: Click once on feature center
   - **BBox**: Click and drag around feature
3. **Set confidence threshold** (slider/spinner: 0.01-0.99)
4. **Name your feature** (e.g., "fish", "ball", "hand")
5. **Navigate to start frame** (where feature first appears)
6. **Click or drag** on the feature
7. **Wait for tracking** (progress bar shows status)
8. **Review result** — feature appears in list with frame range

### Confidence Threshold Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.7-0.9** | Stops quickly | Clean features, avoid false positives |
| **0.5-0.7** | Balanced | Default, good for most videos |
| **0.3-0.5** | Tracks longer | Motion blur, partial occlusion |
| **0.1-0.3** | Aggressive | Challenging scenes, may include errors |

**Rule of thumb:** Start at 0.5. If tracking stops too early, lower it. If segmentation becomes inaccurate, raise it.

### Managing Tracked Features

- **Delete**: Select feature in list, click "Delete"
- **Rename**: Type new name in "Name" field, select feature, click "Rename"
- **Color-coded**: Each feature gets a unique color overlay
- **Frame range**: Shows as "name (f10–f95)" in list

### Export Options

- **Export Video**: Renders MP4 with mask overlays
- **Save JSON**: Exports tracking metadata (bboxes, frame ranges, etc.)

---

## 📊 JSON Output Format

```json
{
  "video_path": "frames/",
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
        "1": [582, 322, 702, 402]
      }
    }
  ]
}
```

**Note:** Masks are NOT saved in JSON (too large). Use exported video or modify code to save as PNG sequences.

---

## 🔧 Extracting Frames with ffmpeg

### Basic Extraction

```bash
# High quality (recommended)
ffmpeg -i video.mp4 -q:v 2 -start_number 0 frames/%05d.jpg

# Lower quality (smaller files, faster)
ffmpeg -i video.mp4 -q:v 5 -start_number 0 frames/%05d.jpg
```

### Advanced Options

```bash
# Extract every 2nd frame (reduce frame count)
ffmpeg -i video.mp4 -vf "select=not(mod(n\,2))" -vsync 0 -q:v 2 -start_number 0 frames/%05d.jpg

# Extract specific time range (10-20 seconds)
ffmpeg -i video.mp4 -ss 00:00:10 -to 00:00:20 -q:v 2 -start_number 0 frames/%05d.jpg

# Resize frames (faster tracking)
ffmpeg -i video.mp4 -vf "scale=960:-1" -q:v 2 -start_number 0 frames/%05d.jpg
```

### Using the Helper Script

```bash
# Extract with default naming
./extract_frames.sh examples/01_dog.mp4
# Creates: 01_dog_frames/

# Custom output directory
./extract_frames.sh examples/01_dog.mp4 my_frames
# Creates: my_frames/
```

---

## 🐛 Troubleshooting

### "decord not installed" (when using video files)

**Solution:** Use frame directories instead:
```bash
./extract_frames.sh video.mp4
python run_tracker.py video_frames/ --frames-dir
```

Or install decord (requires Python 3.8-3.10):
```bash
pip install eva-decord
```

### "No JPEG files found"

Check frame naming:
```bash
ls frames/*.jpg | head -3
# Should show: 00000.jpg, 00001.jpg, 00002.jpg
```

Frames must be:
- JPEG format (`.jpg` or `.jpeg`)
- Numerically named (e.g., `00000.jpg`, `00001.jpg`)
- Sorted by filename

### Tracking stops immediately

**Causes:**
1. Confidence too high → Lower to 0.2-0.4
2. Poor initial click → Click more precisely on feature center
3. Feature occluded/off-screen → Initialize on clearer frame

### Tracking continues with wrong mask

**Causes:**
1. Confidence too low → Raise to 0.6-0.8
2. Feature left frame → Model may track background

### Slow performance

**Solutions:**
- Extract fewer frames (every 2nd or 5th frame)
- Resize frames before tracking (use ffmpeg scale filter)
- Close other applications
- Try `--device cpu` if MPS has issues

---

## 🏗️ How It Works

1. **User clicks/drags** → Initialize tracking point or bbox
2. **EdgeTAM segments** → Pixel-accurate mask on init frame
3. **Propagate forward** → EdgeTAM tracks through subsequent frames
4. **Confidence check** → For each frame:
   - Compute `conf = sigmoid(max_logit)`
   - If `conf < threshold` → **STOP**
   - Else → continue tracking
5. **Add to store** → Save masks, bboxes, metadata
6. **Export** → Render video or save JSON

### Confidence Calculation

```python
logits = model_output  # Raw segmentation logits
mask = (logits > 0.0)  # Binary mask

max_logit = logits.max()
confidence = 1.0 / (1.0 + exp(-max_logit))  # Sigmoid

if confidence < threshold:
    stop_tracking()
```

---

## 📁 Project Structure

```
EdgeTAM/
├── run_tracker.py              # Main tracker script (NEW)
├── extract_frames.sh           # Frame extraction helper (NEW)
├── QUICKSTART.md               # Quick start guide (NEW)
├── TRACKER_README.md           # This file (NEW)
├── TRACKER_INSTALL.md          # Full installation guide (NEW)
├── gradio_app.py               # Original Gradio demo
├── checkpoints/
│   └── edgetam.pt              # Model checkpoint (54 MB)
├── sam2/
│   ├── build_sam.py
│   ├── sam2_video_predictor.py # EdgeTAM video API
│   └── configs/
│       └── edgetam.yaml
└── examples/                   # 24 example videos
    ├── 01_dog.mp4
    ├── 02_cups.mp4
    └── ...
```

---

## ⚙️ Performance

**On Apple Silicon:**
- Device: MPS (Metal Performance Shaders)
- Tracking speed: 5-15 FPS (depends on resolution)
- Memory: ~2-4 GB for 720p
- Model size: 56 MB

**Optimizations:**
- Lightweight RepViT backbone (22× faster than SAM 2)
- Video frames offloaded to CPU (saves GPU memory)
- Background threading (non-blocking UI)

---

## 📝 Comparison: run_api.py vs. run_tracker.py

| Feature | run_api.py | run_tracker.py |
|---------|------------|----------------|
| **Purpose** | Manual bbox annotation | Auto feature tracking |
| **Interaction** | Draw bbox per frame | Click once, auto-propagate |
| **Segmentation** | Bboxes only | Pixel-accurate masks |
| **Model** | None | EdgeTAM (SAM 2) |
| **Init mode** | BBox only | Point OR BBox |
| **Tracking** | Manual | Automatic w/ confidence |

---

## 🎓 Next Steps

1. **Extract frames** from a test video:
   ```bash
   ./extract_frames.sh examples/01_dog.mp4
   ```

2. **Run the tracker**:
   ```bash
   python run_tracker.py 01_dog_frames/ --frames-dir
   ```

3. **Try tracking**:
   - Enable Tracker Mode
   - Set confidence to 0.5
   - Click on the dog
   - Watch it track automatically!

4. **Experiment**:
   - Try different confidence values (0.2 - 0.8)
   - Track multiple features in one video
   - Export and review the results

---

## 📚 Additional Documentation

- **[QUICKSTART.md](QUICKSTART.md)** — Fast setup without decord
- **[TRACKER_INSTALL.md](TRACKER_INSTALL.md)** — Full installation guide with troubleshooting
- **[EdgeTAM README](README.md)** — Original EdgeTAM documentation

---

## 💡 Tips & Tricks

- **Precise clicking**: Click on the center/most visible part of the feature
- **BBox for complex shapes**: Use BBox mode if feature has irregular shape
- **Start on clear frames**: Initialize on frames where feature is fully visible
- **Multiple passes**: Track the same feature multiple times with different thresholds
- **Preview before export**: Use playback controls to review tracking before exporting

---

## 🙏 Credits

- **EdgeTAM**: Meta AI Research (optimized SAM 2 for edge devices)
- **GUI Design**: Based on run_api.py from ctag_api project
- **PyQt Compatibility**: Supports both PyQt5 and PyQt6

---

Enjoy tracking! 🎯
