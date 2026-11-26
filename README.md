<div align="center">

# Vehicle Tracking (OpenCV)

**Detect, track and time every car in a traffic video, with classic computer vision only.**

Background subtraction, morphology and a centroid tracker give each car a stable ID, find the moment it is closest to the camera, and export an annotated video plus a JSON report. No deep learning, no GPU, no model to download: a single Python file and OpenCV.

[Demo](#demo) · [How it works](#how-it-works) · [Tech stack](#tech-stack) · [Getting started](#getting-started) · [Documentation](DOCUMENTATION.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B%20%7C%205-5c3ee8?logo=opencv&logoColor=white)
![No ML](https://img.shields.io/badge/Deep_learning-none-f97316)
![Tests](https://img.shields.io/badge/Tests-26_passing-22c55e)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

## Demo

<p align="center">
  <img src="docs/demo.gif" alt="Cars crossing the frame, each one boxed in green with a stable ID" width="80%" />
</p>

<p align="center">
  <img src="docs/screenshots/car-2-closest.jpg" alt="Car 2, a white van, at its closest moment" width="32%" />
  <img src="docs/screenshots/car-3-far-lane.jpg" alt="Car 3, a dark car in the far lane, tracked as it enters" width="32%" />
  <img src="docs/screenshots/car-4-closest.jpg" alt="Car 4, in the near lane, at its closest moment" width="32%" />
</p>

On the sample video (`cars.mp4`, 1920×1080, 30 fps, 607 frames), the script finds **5 vehicles, 5 IDs, no false positive and no ID switch**:

| ID | Seen | Direction | Closest moment | Bounding box (x, y, w, h) |
| --: | --- | --- | --: | --- |
| 1 | 0.3 s → 1.8 s | ← left | 1.37 s | (0, 367, 590, 236) ¹ |
| 2 | 3.2 s → 4.8 s | ← left | 4.27 s | (171, 355, 534, 224) |
| 3 | 4.6 s → 6.2 s | ← left | 5.63 s | (330, 402, 420, 127) ² |
| 4 | 6.7 s → 8.1 s | → right | 7.07 s | (19, 398, 912, 347) |
| 5 | 14.6 s → 16.5 s | ← left | 16.00 s | (23, 405, 531, 194) |

¹ Car 1 tows a trailer: both are counted as one vehicle, and the box includes the trailer.
² Car 3 is a dark car in the far lane, in front of dark trees: it is tracked over its whole crossing, but only part of it stands out from the background, so its box is smaller than the car.

The matching JSON output is in [`docs/sample-results.json`](docs/sample-results.json) (OpenCV 5.1; other OpenCV versions can shift a frame or two).

### Robustness

The same video, altered to leave the tuning conditions. Expected: 5 cars.

| Variant | 720p | 480p | 4K | Light flashes | Camera jolt |
| --- | :-: | :-: | :-: | :-: | :-: |
| Cars found | 5 | 5 | 4 / 4 ³ | 5 | 5 |

³ First 260 frames only, which contain 4 cars. Results are also identical with OpenCV 4.13 and 5.1.

## Features

- **Motion detection** with OpenCV's MOG2 background subtractor, which learns the empty road online, with no training data.
- **Shadow removal**: MOG2 marks shadows in gray, and a threshold drops them before they turn into false detections.
- **Flash and jolt guard**: when more than half of the frame changes at once (lighting, camera shake), the frame is skipped instead of producing fake cars.
- **Any resolution**: detection runs on a copy resized to 1080p, so the tuned values work the same from 480p to 4K, and boxes are drawn at the real size.
- **Mask cleanup**: a morphological opening removes noise, then dilation and erosion stitch the fragments of one car back into a single blob.
- **Box filtering and merging**: contours are filtered by area, size and aspect ratio, then nearby boxes are merged (up to 200 px apart horizontally, 100 px vertically) so one vehicle, even with a trailer, gives one box.
- **Stable IDs**: global nearest-first matching (independent of detection order) with confirmed cars served first, motion prediction for fast cars, and a 0.4 s grace period. IDs are numbered 1, 2, 3… with no gaps.
- **Only things that travel become cars**: a track is confirmed after 3 detections *and* 75 px of travel, so foliage moving in the wind never gets an ID, which lets the size filter accept small cars in the far lane.
- **Ground-contact tracking**: each car is followed by the bottom-centre of its box (wheels and shadow), which stays put when the top of a dark car blends into the background.
- **Per-car report**: closest moment (largest box), first and last time seen, and direction of travel.
- **Outputs**: an annotated video, a JSON report and a summary table in the terminal.
- **Live preview** at the video's real speed, in two windows: the annotated video and the cleaned mask. `q`, `Esc` or closing the window stops it, and `Ctrl+C` still saves what was processed.
- **Command line**: choose the input and outputs, or run headless at full speed.

## How it works

Each frame goes through a linear pipeline:

```text
Frame ─► Resize to 1080p ─► Gaussian blur ─► MOG2 ─► Threshold (drop shadows)
      ─► Global change? (skip) ─► Open / Dilate / Erode ─► Contours + filters
      ─► Merge nearby boxes ─► Tracking (IDs, must travel) ─► Closest moment ─► Video + JSON
```

| Step | Technique | Why |
| --- | --- | --- |
| Normalize | Resize so `min(w, h) = 1080` | All pixel thresholds stay valid at any resolution |
| Denoise | `GaussianBlur` 11×11 | Smooths camera noise before subtraction |
| Detect motion | MOG2 (`history=700`, `varThreshold=120`) | Robust built-in subtractor with shadow detection |
| Ignore global changes | Skip if > 50 % of the frame is foreground | A flash or a jolt is not a car |
| Clean the mask | Threshold at 200, then open / dilate / erode | Removes shadows and specks, joins car fragments |
| Keep car-sized blobs | Area ≥ 6000 px², `w ≥ 40`, `h ≥ 95`, aspect 0.3 to 4 | Rejects small or oddly shaped blobs |
| One vehicle, one box | Merge boxes closer than 200 px horizontally / 100 px vertically | A car (or car + trailer) can be split into several contours |
| Track | Bottom-centre point, confirmed cars first, closest pairs first, predicted or last position, within 350 px | Stable, order-independent, follows fast cars |
| Confirm | 3 detections and 75 px travelled | Cars cross the frame; foliage stays put |
| Closest moment | Largest bounding-box area per ID | Things look bigger when they are closer |

All tuning values are constants at the top of [`app.py`](app.py). The full write-up (design choices, accuracy, robustness tests, strengths, weaknesses and next steps) is in [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Tech stack

| Area | Choice |
| --- | --- |
| Language | Python 3.10+ (tested with Python 3.14) |
| Computer vision | OpenCV (`opencv-python`, tested with 4.13 and 5.1) |
| Standard library | `argparse`, `dataclasses`, `pathlib`, `json`, `math`, `time` |
| Tests | `unittest` (standard library), 26 tests including end-to-end runs on generated videos |
| Video codec | MPEG-4 (`mp4v`) through `cv2.VideoWriter` |

## Getting started

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Nathan-Part/vehicle-tracking-opencv.git
cd vehicle-tracking-opencv
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install OpenCV and run:

```bash
pip install -r requirements.txt
python app.py
```

Two windows open, **Video** (annotated) and **Mask** (foreground mask). Press **`q`** or **`Esc`** to stop early. At the end, the terminal prints a summary of every car, and `output.mp4` and `results.json` are written next to `app.py`.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | `cars.mp4` | Video to analyse |
| `-o`, `--output` | `output.mp4` | Annotated video |
| `-j`, `--json` | `results.json` | JSON report |
| `--no-video` | off | Don't write the annotated video |
| `--no-display` | off | No preview windows, process as fast as possible (servers, CI) |

```bash
python app.py --input my_video.mp4 --output out/annotated.mp4 --json out/report.json
python app.py --no-display --no-video     # just the report, as fast as possible
```

Output folders are created automatically, and the script refuses to write the output video over the input.

### Run the tests

```bash
python -m unittest -v
```

### Use your own video

The camera must be **fixed**. The resolution doesn't matter, but the size filters assume cars fill roughly the same share of the frame as in the sample video. For a camera much farther away or at another angle, adjust `MIN_CONTOUR_AREA`, `MIN_BOX_W` / `MIN_BOX_H`, `MERGE_GAP_X` / `MERGE_GAP_Y`, `MIN_MOVE` and `MAX_DISTANCE` at the top of `app.py`.

## Output

| File | Content |
| --- | --- |
| `output.mp4` | The input video with boxes, IDs, the car count and the processing speed |
| `results.json` | One object per car (see below) |

```json
{
  "id": 4,
  "closest_frame": 212,
  "closest_time_sec": 7.07,
  "bbox": [19, 398, 912, 347],
  "area": 316464,
  "first_seen_sec": 6.63,
  "last_seen_sec": 8.0,
  "direction": "right"
}
```

Both files are regenerated on every run, so they are not versioned.

## Limitations

- **Fixed camera only**: short jolts are skipped, but a moving camera sees motion everywhere.
- **Stopped cars disappear**: MOG2 absorbs a stationary car into the background after a while.
- **Low contrast**: a dark car in front of dark trees is still tracked, but only partly detected, so its box can be too small and can blink for a frame or two.
- **Close cars can merge**: two cars less than 200 px apart in the same lane end up in one box.
- **No re-identification**: a car lost for more than 0.4 s gets a new ID.

## Project structure

```text
.
├── app.py                 # The whole pipeline: detection, tracking, report, CLI
├── cars.mp4               # Sample input video
├── DOCUMENTATION.md       # Technical write-up and critical analysis
├── requirements.txt       # opencv-python
├── tests/
│   └── test_app.py        # Unit and end-to-end tests
├── docs/
│   ├── demo.gif           # Animated preview used in this README
│   ├── sample-results.json
│   └── screenshots/       # Cars 2, 3 and 4
└── LICENSE
```

## Author

Created by Nathan Partouche.

## License

[MIT](LICENSE)
