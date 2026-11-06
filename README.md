# Aerial Vehicle Detection

A classical computer-vision pipeline that detects and tracks moving vehicles in aerial / overhead video using OpenCV - no deep learning. It combines colour and motion cues to isolate moving cars, and includes an experimental size-based altitude estimate.

> Academic project built with OpenCV. This is classical computer vision (colour, motion, contours), not a machine-learning detector.

## What it does

- Reads a video stream frame by frame (OpenCV `VideoCapture`).
- Isolates candidate objects with HSV colour masking plus morphological cleanup.
- Detects motion between consecutive frames with Farnebäck optical flow and intersects it with the colour mask, so only *moving* regions survive.
- Finds contours and keeps those within a car-like area and width/height ratio, then draws bounding boxes and centre points.
- Optionally estimates a rough altitude from the apparent size of detected objects (see [Altitude estimate](#altitude-estimate)).

## Tech stack

- **Python**
- **OpenCV** (`opencv-python`) HSV filtering, optical flow, contours, morphology
- **NumPy**
- **Matplotlib**

## Getting started

### Prerequisites

- Python 3.9+

### Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

No sample video is bundled. Place a video at `data/sample_video.mp4` (relative to the working directory), then run from the `src` directory so the module imports resolve:

```bash
cd src
python main.py
```

Press `q` to quit. The window shows the annotated frame next to the binary mask.

## Altitude estimate

`AltitudeAnalyzer` estimates altitude from the average apparent size of detected objects, assuming an inverse-square relationship between object size and altitude, anchored to a single hardcoded reference (≈500 px at 100 m). It is a heuristic for experimentation, **not a calibrated measurement** it assumes a fixed real-world object size and a nadir (straight-down) camera, with no camera calibration.

## Project structure

```
src/
├── main.py                   # Video loop: capture → detect → analyze → display
└── utils/
    ├── detector.py           # ObjectDetector: HSV + optical flow + contour filtering
    └── altitude_analyzer.py  # AltitudeAnalyzer: size-based altitude heuristic
requirements.txt
```

## Limitations & possible improvements

- **Classical CV, not learned.** Detection relies on colour, motion and shape heuristics, so it is sensitive to lighting, background and camera motion. A learned detector (e.g. YOLO) would be far more robust and would also classify object types.
- **Permissive colour mask.** The HSV range is wide, so most of the actual filtering comes from motion and shape, stationary vehicles are not detected.
- **Altitude estimate is uncalibrated** (see above) and should be treated as experimental.
- **Hardcoded paths.** The input video path and working-directory assumptions are fixed; adjust them for your setup.
- **No sample data** is included in the repository.
