# Vehicle Detection & Tracking System — Technical Documentation

## 1. Behind the Scenes: Logic, Algorithms & Data Flow

### Architecture Overview

The system is a **single-file Python script** (`app.py`) that processes a video frame-by-frame using only OpenCV, with no external ML frameworks. It follows a linear 6-step pipeline:

```
Video Frame
    │
    ▼
┌──────────────────────┐
│  1. Gaussian Blur     │  Reduce camera noise
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  2. Background        │  MOG2 subtraction → foreground mask
│     Subtraction       │
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
│  5. Centroid Tracking │  Match detections to previous objects
│     (ID assignment)   │  by nearest center (Euclidean distance)
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
| **Motion detection** | MOG2 (Mixture of Gaussians) | Built into OpenCV, handles gradual lighting changes, supports shadow detection |
| **Noise removal** | Morphological OPEN (erode→dilate) | Removes small noise dots without affecting large blobs |
| **Fragment merging** | Morphological DILATE + ERODE | Bridges gaps between fragments of the same car |
| **Box merging** | Iterative pairwise merge (MERGE_GAP=100px) | Simple to implement and explain; consolidates fragmented detections |
| **Tracking** | Nearest-centroid matching | Beginner-friendly; no complex state estimation needed |
| **Closest moment** | Max bounding-box area | Larger box = closer to camera (perspective geometry) |

### Data Flow Per Frame

1. **Input**: Raw BGR frame from `cv2.VideoCapture`
2. **Blur**: `GaussianBlur(11×11)` smooths pixel-level noise
3. **MOG2**: Outputs a grayscale mask (white=foreground, gray=shadow, black=background)
4. **Threshold**: Binary mask — removes shadows (gray→black)
5. **Morphology**: OPEN removes noise, DILATE merges fragments, ERODE shrinks back to natural size
6. **Contours**: `findContours(RETR_EXTERNAL)` extracts outer boundaries
7. **Filtering**: Reject contours with area < 6000px², height < 130px, or aspect ratio outside [0.3, 4.0]
8. **Merging**: Iteratively merge boxes within 100px of each other
9. **Tracking**: Compare each detection's center to all known objects; assign existing ID if distance < 350px, else assign new ID
10. **Output**: Draw boxes + IDs on frame, update `closest_moment` dict, write to video file

---

## 2. Design Decisions: Why This Way?

| Decision | Rationale |
|---|---|
| **Single file** | Easy to read, run, and explain in an interview setting |
| **No deep learning** | Only standard library + OpenCV required; no GPU, no model downloads |
| **MOG2 over MOG/KNN** | MOG2 is the most robust built-in background subtractor; handles multi-modal backgrounds |
| **Shadow detection ON** | Shadows cause false detections; MOG2 marks them as gray so we can threshold them out |
| **Aggressive MERGE_GAP (100px)** | A single car can produce multiple contours; merging prevents duplicate IDs |
| **MIN_HITS = 3** | Objects must be seen 3 consecutive frames before drawing — eliminates single-frame false positives |
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
| Real cars observed | 5 |
| IDs assigned | 5 |
| Correct detections | 5/5 (100%) |
| False positives | 0 |
| ID splits (same car → 2 IDs) | 0 |
| ID merges (2 cars → 1 ID) | 0 |

### Caveats

- This is measured on a **single test video** with a fixed camera angle
- Accuracy will vary with different lighting, camera positions, and traffic density
- No formal metrics (precision/recall/F1) were computed — evaluation was purely visual
- The system was iteratively tuned to perform well on this specific video

---

## 4. Critical Analysis: Strengths & Weaknesses

### Strengths

| Strength | Detail |
|---|---|
| **Simplicity** | ~240 lines, 4 imports, no external ML frameworks |
| **Real-time capable** | Processes each frame fast enough for 30fps playback |
| **No training required** | MOG2 learns the background online — no dataset needed |
| **Robust to shadows** | Shadow detection + thresholding eliminates a common source of false positives |
| **Stable IDs** | Grace period (MAX_UNSEEN) + MIN_HITS + box merging produce consistent tracking |
| **Complete pipeline** | Detection → Tracking → Analysis → Export (JSON + annotated video) |

### Weaknesses

| Weakness | Detail |
|---|---|
| **Stationary camera only** | Background subtraction requires a fixed camera; any camera movement would break detection |
| **Cannot detect stopped cars** | A parked car becomes part of the "background" after MOG2's history window |
| **Small/distant cars missed** | The height filter (h < 130px) rejects small objects — necessary to avoid false positives but limits range |
| **No occlusion handling** | When cars overlap, they may merge into a single detection or swap IDs |
| **Parameter-dependent** | Tuned specifically for this video; different scenes (night, rain, different angles) would require re-tuning |
| **Greedy matching** | The nearest-centroid tracker can make suboptimal assignments when cars are close together |
| **No re-identification** | Once a car's ID is lost, it cannot be recovered — the car gets a new ID |

---

## 5. If I Had One Extra Hour

I would prioritize in this order:

### Priority 1: Replace centroid tracking with a Kalman Filter (~30 min)

**Why**: The current tracker only uses position. A Kalman filter would **predict** where each car will be in the next frame based on its velocity. This would:
- Dramatically reduce ID switches when detection temporarily fails
- Handle fast-moving cars more reliably
- Improve tracking through brief occlusions

**Impact**: High — this is the single biggest improvement for tracking stability.

### Priority 2: Add an IoU-based (Intersection over Union) matching step (~15 min)

**Why**: Currently, matching uses only the center point distance. IoU-based matching would compare the **overlap** between predicted and detected bounding boxes, which is more robust when cars are close together or have similar centers but different sizes.

**Impact**: Medium — reduces ID swaps between nearby cars.

### Priority 3: Automated accuracy measurement (~15 min)

**Why**: Currently, accuracy is measured by visual inspection. Adding a simple ground-truth annotation file (JSON with frame ranges and expected car counts) would allow automated precision/recall computation. This would make parameter tuning systematic instead of trial-and-error.

**Impact**: Medium — enables faster iteration and objective quality measurement.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `cv2` (OpenCV) | 4.x | Video I/O, image processing, drawing |
| `math` | stdlib | Euclidean distance (`hypot`) |
| `json` | stdlib | Export results |
| `sys` | stdlib | Error handling (`sys.exit`) |

## Output Files

| File | Description |
|---|---|
| `results.json` | Closest moment per car (ID, frame, timestamp, bbox, area) |
| `output.mp4` | Annotated video with bounding boxes and ID labels |
