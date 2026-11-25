# Vehicle Detection & Tracking System — Technical Documentation

## 1. Behind the Scenes: Logic, Algorithms & Data Flow

### Architecture Overview

The system is a **single Python file** (`app.py`) that processes a video frame by frame using only OpenCV, with no ML framework. The code is split into small, testable pieces:

| Piece | Role |
|---|---|
| `Detector` | Resize to the working size, blur, MOG2, mask cleanup, contours, filters, box merging |
| `merge_boxes()` | Merges overlapping or nearby boxes so one car gives one box |
| `CentroidTracker` | Gives each car a stable ID across frames |
| `CarStats` / `build_report()` | Closest moment, first and last sighting, direction |
| `process_video()` | Main loop: read, detect, track, draw, write, display |
| `main()` | Command-line interface (`--input`, `--output`, `--json`, `--no-video`, `--no-display`) |

```
Video Frame
    │
    ▼
┌──────────────────────┐
│  0. Working size      │  Resize so min(w, h) = 1080 px
└──────────┬───────────┘  (all pixel thresholds are tuned for 1080p)
           ▼
┌──────────────────────┐
│  1. Gaussian Blur     │  Reduce camera noise
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  2. Background        │  MOG2 subtraction → foreground mask
│     Subtraction       │  > 50 % of the frame moving → global change, frame skipped
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  3. Mask Cleanup      │  Threshold + Morphological ops
│     (OPEN/DILATE/     │  (remove noise, fill gaps,
│      ERODE)           │   merge fragments)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  4. Contour Detection │  findContours → bounding boxes
│     + Filtering       │  Filter by area, size, aspect ratio
│     + Box Merging     │  Merge nearby boxes (MERGE_GAP=100px)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  5. Centroid Tracking │  Global nearest-first matching against the
│     (ID assignment)   │  predicted or last position of each car
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  6. Closest Moment    │  Track largest bbox area per ID
│     + JSON Export     │  Export results + annotated video
└──────────────────────┘
```

### Key Algorithms

| Component | Algorithm | Why This Choice |
|---|---|---|
| **Resolution independence** | Detection on a copy resized to 1080p, boxes mapped back | The tuned thresholds stay valid for 480p, 720p or 4K |
| **Motion detection** | MOG2 (Mixture of Gaussians) | Built into OpenCV, handles gradual lighting changes, supports shadow detection |
| **Global change guard** | Skip the frame if > 50 % of the pixels are foreground | A lighting flash or camera jolt is not a car |
| **Noise removal** | Morphological OPEN (erode→dilate) | Removes small noise dots without affecting large blobs |
| **Fragment merging** | Morphological DILATE + ERODE | Bridges gaps between fragments of the same car |
| **Box merging** | Iterative pairwise merge (MERGE_GAP=100px) | Simple to implement and explain; consolidates fragmented detections |
| **Tracking** | Nearest-centroid, global nearest-first matching, constant-velocity prediction | Order-independent, follows fast cars, still beginner-friendly |
| **Closest moment** | Max bounding-box area | Larger box = closer to camera (perspective geometry) |

### Data Flow Per Frame

1. **Input**: Raw BGR frame from `cv2.VideoCapture`
2. **Resize**: to the working size where `min(width, height) = 1080` (no-op for a 1080p video)
3. **Blur**: `GaussianBlur(11×11)` smooths pixel-level noise
4. **MOG2**: Outputs a grayscale mask (white=foreground, gray=shadow, black=background)
5. **Threshold**: Binary mask — removes shadows (gray→black)
6. **Global change check**: if more than half the frame is foreground, no detection this frame (tracks just age)
7. **Morphology**: OPEN removes noise, DILATE merges fragments, ERODE shrinks back to natural size
8. **Contours**: `findContours(RETR_EXTERNAL)` extracts outer boundaries
9. **Filtering**: Reject contours with area < 6000px², width < 40px, height < 130px, or aspect ratio outside [0.3, 4.0]
10. **Merging**: Iteratively merge boxes within 100px of each other, then map boxes back to the video's real size
11. **Tracking**: Every (detection, car) pair closer than 350px (to the car's predicted *or* last position) is ranked by distance; the closest pairs are matched first. Unmatched detections start new tracks
12. **Output**: Draw boxes + IDs on frame, update per-car statistics, write to video file

### Tracking details

- **Prediction**: each track keeps a smoothed velocity (`VELOCITY_SMOOTHING = 0.5`). Its expected position is `last position + velocity × frames since last seen`.
- **Predicted or last position**: the distance used for matching is the smaller of the two. The prediction helps fast cars; the last position keeps working when the velocity estimate is wrong, for example when a car leaves the frame and its box shrinks. On `cars.mp4`, using only the prediction split the first car and the trailer it tows into two IDs.
- **Confirmation**: a track becomes a car after `MIN_HITS = 3` detections (cumulative, not necessarily consecutive). Only then does it get a public ID, so IDs are 1, 2, 3… with no gaps caused by one-frame noise.
- **Grace period**: a track not seen for more than `MAX_UNSEEN = 8` frames is dropped.

---

## 2. Design Decisions: Why This Way?

| Decision | Rationale |
|---|---|
| **Single file** | Easy to read, run, and explain in an interview setting |
| **No deep learning** | Only OpenCV required; no GPU, no model downloads |
| **Paths relative to the script + CLI** | Works wherever the project is cloned; any video can be processed without editing the code |
| **Resize to a working size** | Scaling each threshold separately breaks on rounding (a 5 px kernel at 720p becomes 3 px instead of 3.3); resizing the frame keeps every value exactly as tuned |
| **MOG2 over MOG/KNN** | MOG2 is the most robust built-in background subtractor; handles multi-modal backgrounds |
| **Shadow detection ON** | Shadows cause false detections; MOG2 marks them as gray so we can threshold them out |
| **Global change guard (50 %)** | The largest car on the test video covers about 15 % of the frame, so 50 % only triggers on lighting changes and camera jolts |
| **Aggressive MERGE_GAP (100px)** | A single car can produce multiple contours; merging prevents duplicate IDs |
| **MIN_HITS = 3** | A track needs 3 detections before it is drawn — eliminates single-frame false positives |
| **MAX_UNSEEN = 8** | Grace period of ~0.27s at 30fps — keeps IDs stable during brief detection gaps without causing ID reuse by different cars |
| **MAX_DISTANCE = 350** | Cars on a 1920px-wide video can move fast; a generous threshold prevents the same car from getting a new ID |
| **Closest = largest bbox** | Simple proxy for "closest to camera" based on perspective: objects appear larger when closer |

---

## 3. System Accuracy & Measurement

### How Accuracy Was Measured

Accuracy was evaluated through **manual visual inspection** of the test video (`cars.mp4`, 607 frames, 1920×1080, 30fps):

- Counted the number of **real cars** passing through the scene
- Compared against the number of **unique IDs** assigned by the system
- Checked for **false positives** (non-car objects receiving an ID)
- Checked for **ID splits** (same car getting multiple IDs)
- Checked for **ID merges** (different cars sharing the same ID)

### Results on Test Video

| Metric | Value |
|---|---|
| Real vehicles observed | 5 (the first one tows a trailer) |
| IDs assigned | 5 |
| Correct detections | 5/5 (100%) |
| False positives | 0 |
| ID splits (same car → 2 IDs) | 0 |
| ID merges (2 cars → 1 ID) | 0 |

### Robustness tests

The test video was altered to check how the system behaves outside its tuning conditions (expected result: 5 cars every time).

| Variant | Before the refactor | Now |
|---|---|---|
| Original 1080p | 5 | 5 |
| Downscaled to 720p | 4 (one car missed) | 5 |
| Downscaled to 480p | 1 | 5 |
| Upscaled to 4K (first 260 frames, 4 cars) | — | 4 |
| Light flashes (4 frames, +90 brightness) | 6 (one false car) | 5 (4 frames skipped) |
| Camera jolt (25 px, 4 frames) | 5 | 5 |

### Automated tests

`tests/test_app.py` (standard `unittest`, run with `python -m unittest -v`) covers box merging, the working scale, the tracker (confirmation, grace period, no ID gaps, order independence, motion prediction), the JSON report, and end-to-end runs on generated videos (two cars in opposite directions, a light flash, an unreadable file, CLI errors).

### Caveats

- The car count is measured on a **single real video** with a fixed camera angle; the robustness tests are variations of that same video
- Accuracy will vary with different lighting, camera positions, and traffic density
- No formal metrics (precision/recall/F1) were computed on annotated ground truth
- The system was iteratively tuned to perform well on this specific video

---

## 4. Critical Analysis: Strengths & Weaknesses

### Strengths

| Strength | Detail |
|---|---|
| **Simplicity** | One file, OpenCV only, no external ML framework |
| **Real-time capable** | About 16 to 19 fps on a 1080p video on a laptop CPU (headless, video writing included); the preview plays at the video's real speed |
| **No training required** | MOG2 learns the background online — no dataset needed |
| **Robust to shadows and flashes** | Shadow thresholding plus the global change guard |
| **Resolution independent** | Same results from 480p to 4K without retuning |
| **Stable IDs** | Grace period + MIN_HITS + box merging + motion prediction + order-independent matching |
| **Complete pipeline** | Detection → Tracking → Analysis → Export (JSON + annotated video), with a CLI and tests |

### Weaknesses

| Weakness | Detail |
|---|---|
| **Stationary camera only** | Background subtraction requires a fixed camera; continuous camera movement breaks detection (short jolts are skipped) |
| **Cannot detect stopped cars** | A parked car becomes part of the "background" after MOG2's history window |
| **Small/distant cars missed** | The height filter (h < 130px at 1080p) rejects small objects — necessary to avoid false positives but limits range |
| **Low-contrast cars** | A dark car in front of dark trees is only partly detected, so its box can cover only part of the car |
| **No occlusion handling** | When cars overlap, they may merge into a single detection or swap IDs |
| **Scene-dependent** | Resolution no longer matters, but distance, angle, night or rain would still require re-tuning |
| **No re-identification** | Once a car's ID is lost for more than 8 frames, it gets a new ID |
| **"Closest" is a proxy** | The largest box can include the shadow, or two merged cars; on a side view where all cars are at a similar distance, it mostly picks the frame where the box is largest |

---

## 5. Next Steps

Done since the first version: motion prediction, order-independent matching, resolution independence, global change guard, CLI, automated tests.

### Priority 1: Kalman filter (~30 min)

Replace the smoothed constant-velocity estimate with `cv2.KalmanFilter`, which also models uncertainty. This would make the prediction more reliable through longer occlusions.

### Priority 2: IoU-based matching (~15 min)

Compare the **overlap** between predicted and detected boxes instead of only centre distances. More robust when cars are close together or have similar centres but different sizes.

### Priority 3: Ground-truth evaluation (~15 min)

Annotate the test video (frame ranges per car) and compute precision/recall automatically, so parameter tuning becomes systematic instead of visual.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `cv2` (OpenCV) | 4.8+ (tested with 4.13 and 5.1) | Video I/O, image processing, drawing |
| `numpy` | installed with OpenCV | Only used by the tests to generate videos |
| `argparse`, `json`, `math`, `time`, `pathlib`, `dataclasses`, `sys` | stdlib | CLI, export, distances, timing, paths |

## Output Files

| File | Description |
|---|---|
| `results.json` | Per car: ID, closest frame and time, bbox, area, first and last time seen, direction |
| `output.mp4` | Annotated video with bounding boxes, ID labels, car count and processing speed |
