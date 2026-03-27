# media_analyzer

Python CLI app that analyzes **images** and **videos** to detect objects using **YOLOv8 (Ultralytics)** with the CPU-friendly `yolov8n.pt` model (auto-downloads on first run).

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

From the project root:

```bash
pip install -r media_analyzer/requirements.txt
```

### 3) Install FFmpeg (required for videos)

FFmpeg must be installed **system-wide** and available on `PATH`.

- Verify:

```bash
ffmpeg -version
```

If FFmpeg is missing, video analysis will fail with a clear error message.

## Usage

Run from the project root:

### Analyze an image

```bash
python media_analyzer/analyze.py path/to/image.jpg
```

### Analyze a video (extract frames every 1 second)

```bash
python media_analyzer/analyze.py path/to/video.mp4 --interval 1
```

### Save results to JSON

```bash
python media_analyzer/analyze.py path/to/video.mp4 --output results.json
```

### Save extracted/analyzed frames

```bash
python media_analyzer/analyze.py path/to/video.mp4 --save-frames
```

### Draw bounding boxes

```bash
python media_analyzer/analyze.py path/to/image.jpg --draw-boxes
python media_analyzer/analyze.py path/to/video.mp4 --draw-boxes --save-frames
```

## CLI flags

- `file` (positional): input file path (image or video)
- `--interval INT`: frame extraction interval in seconds (default: `1`)
- `--output FILE`: save results to a JSON file
- `--save-frames`: save analyzed frames as images
- `--draw-boxes`: draw bounding boxes on detected objects (saves `*_boxed.*`)
- `--no-progress`: disable progress indicator

## Supported formats

- Images: `jpg`, `jpeg`, `png`, `webp`
- Videos: `mp4`, `mov`, `mkv`

## Example output (console)

```text
Media Analyzer
--------------
File: video.mp4
Type: video
Frames analyzed: 10

Detected objects:
  person        (0.93)
  car           (0.88)
  dog           (0.79)

Results saved to: results.json
```

## JSON output format (`--output`)

```json
{
  "file": "video.mp4",
  "type": "video",
  "frames_analyzed": 10,
  "objects": [
    {"label": "person", "confidence": 0.93},
    {"label": "car", "confidence": 0.88}
  ]
}
```

