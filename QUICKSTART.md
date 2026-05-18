# Video Annotator — Quick Start (No Decord Required)

Use **frame directories** instead of video files. This avoids the optional
`decord` dependency and works with Python 3.11.

## Setup

```bash
git clone https://github.com/jadenvc/video_annotator.git
cd video_annotator

conda create -n video_annotator python=3.11 -y
conda activate video_annotator

pip install torch torchvision
pip install -e ".[tracker]"
bash checkpoints/download_ckpts.sh
```

## Two-Step Process

### Step 1: Extract Frames with ffmpeg

```bash
# Create frames directory
mkdir frames

# Extract JPEG frames from your video
ffmpeg -i your_video.mp4 -q:v 2 -start_number 0 frames/%05d.jpg
```

**What this does:**
- `-q:v 2` → High quality JPEG (scale 1-31, lower = better)
- `-start_number 0` → Start numbering from `00000.jpg`
- `%05d.jpg` → 5-digit zero-padded filenames (00000.jpg, 00001.jpg, ...)

**Example output:**
```
frames/
  00000.jpg
  00001.jpg
  00002.jpg
  ...
  00150.jpg
```

### Step 2: Run the Tracker

```bash
python run_tracker.py frames/ --frames-dir
```

That's it. No decord needed.

---

## Full Example

```bash
# Navigate to the repo
cd video_annotator

# Extract frames from example video
mkdir shark_frames
ffmpeg -i examples/01_dog.mp4 -q:v 2 -start_number 0 shark_frames/%05d.jpg

# Run tracker
python run_tracker.py shark_frames/ --frames-dir --confidence 0.5
```

---

## Command-Line Options

```bash
# Basic usage
python run_tracker.py frames_dir/ --frames-dir

# With custom confidence
python run_tracker.py frames_dir/ --frames-dir --confidence 0.3

# Force CPU (if MPS has issues)
python run_tracker.py frames_dir/ --frames-dir --device cpu

# Force MPS (default on M2)
python run_tracker.py frames_dir/ --frames-dir --device mps
```

---

## GUI Usage (Same as Before)

1. **Enable Tracker Mode** checkbox
2. **Choose Point or BBox** init mode
3. **Set confidence threshold** (0.01-0.99)
4. **Click or drag** on a feature to start tracking
5. Tracker automatically propagates until confidence drops
6. **Export** as video or JSON

---

## Troubleshooting

### "No JPEG files found in directory"

Make sure your frames are named with `.jpg` or `.jpeg` extension:
```bash
ls frames/*.jpg | head -5
```

Should show:
```
frames/00000.jpg
frames/00001.jpg
frames/00002.jpg
```

### "Cannot read first frame"

Check that the first file is valid:
```bash
file frames/00000.jpg
```

Should show: `JPEG image data`

### Extract frames at different quality

```bash
# Higher quality (larger files)
ffmpeg -i video.mp4 -q:v 1 -start_number 0 frames/%05d.jpg

# Lower quality (smaller files, faster)
ffmpeg -i video.mp4 -q:v 5 -start_number 0 frames/%05d.jpg
```

### Extract every Nth frame (for speed)

```bash
# Extract every 2nd frame (half the frames)
ffmpeg -i video.mp4 -vf "select=not(mod(n\,2))" -vsync 0 -q:v 2 -start_number 0 frames/%05d.jpg

# Extract every 5th frame
ffmpeg -i video.mp4 -vf "select=not(mod(n\,5))" -vsync 0 -q:v 2 -start_number 0 frames/%05d.jpg
```

---

## What If I Try to Use a Video File Without Decord?

The script will detect this and show a helpful error:

```
⚠️  WARNING: decord not installed!
Without decord, you must use frame directories instead of video files.

Options:
  1. Extract frames with ffmpeg, then use --frames-dir:
     ffmpeg -i video.mp4 -q:v 2 -start_number 0 frames/%05d.jpg
     python run_tracker.py frames/ --frames-dir

  2. Install decord (requires Python 3.8-3.10):
     pip install eva-decord
```

---

## Performance Notes

**Frame directories vs. Video files:**
- ✅ **Pros**: No decord dependency, works on any Python version
- ⚠️ **Cons**: Takes up more disk space (JPEGs are larger than compressed video)
- ⚡ **Speed**: Similar performance (EdgeTAM loads frames into memory anyway)

**Disk usage example:**
- 1 minute video (30 fps) = 1,800 frames
- At quality `-q:v 2`: ~50-100 KB per frame
- Total: ~90-180 MB for 1 minute

---

## Next Steps

Once you've run the tracker:
1. Check the **Tracked Features** list
2. Use **Export Video** to render with masks
3. Use **Save JSON** to export tracking metadata
4. Experiment with different **confidence thresholds**

Happy tracking! 🎯
