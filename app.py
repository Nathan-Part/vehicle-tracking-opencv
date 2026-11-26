"""Vehicle detection and tracking with classic computer vision (OpenCV only).

Pipeline per frame:
    blur -> MOG2 background subtraction -> drop shadows -> morphology
    -> contours + filters -> merge nearby boxes -> centroid tracking
    -> closest moment per car -> annotated video + JSON report

Usage:
    python app.py                       # cars.mp4 next to this file, with preview
    python app.py --input my_video.mp4  # another video
    python app.py --no-display          # headless (servers, CI), as fast as possible
    python app.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent

# ── Tuning ───────────────────────────────────────────────────────────────────
# Pixel values below were tuned on a 1080p video. Detection runs on a copy of
# each frame resized so that min(width, height) == REF_SIZE, so the same values
# work for 480p, 720p or 4K, and boxes are mapped back to the real size.
REF_SIZE = 1080

BLUR_SIZE = 11             # Gaussian blur kernel (odd)
MOG2_HISTORY = 700
MOG2_VAR_THRESHOLD = 120
SHADOW_THRESHOLD = 200     # MOG2 marks shadows ~127, foreground 255

KERNEL_SMALL = 5           # opening: removes small noise dots
KERNEL_BIG = 15            # dilate/erode: merges fragments of the same car

MIN_CONTOUR_AREA = 6000    # px²
MIN_BOX_W = 40             # px
MIN_BOX_H = 95             # px (cars in the far lane are only ~100 px tall;
                           # parked noise is rejected by MIN_MOVE instead)
MIN_ASPECT, MAX_ASPECT = 0.3, 4.0
MERGE_GAP_X = 200          # merge boxes closer than this horizontally (px):
                           # one vehicle (or a car and its trailer) is long
MERGE_GAP_Y = 100          # ... and vertically (px)

# If more than this share of the frame is "moving", it's a global change
# (lighting flash, camera shake, MOG2 warm-up), not cars: skip detection.
MAX_FOREGROUND_RATIO = 0.5

MAX_DISTANCE = 350         # max centroid jump between frames (px)
MAX_UNSEEN_SEC = 0.4       # how long a lost car is kept in memory (seconds)
MAX_UNSEEN = 12            # same, in frames, at 30 fps (default for the tracker)
MIN_HITS = 3               # detections needed before a car is confirmed
MIN_MOVE = 75              # px a track must travel before it is confirmed:
                           # cars cross the frame, foliage and noise stay put
VELOCITY_SMOOTHING = 0.5   # 0 = ignore motion, 1 = trust last displacement only

FALLBACK_FPS = 30.0


def working_scale(width: int, height: int) -> float:
    """Factor from the video size to the working (reference) size."""
    return REF_SIZE / min(width, height)


# ── Detection ────────────────────────────────────────────────────────────────
def merge_boxes(boxes: list[list[int]], gap_x: float,
                gap_y: float | None = None) -> list[list[int]]:
    """Merge boxes [x1, y1, x2, y2] that overlap or are closer than
    `gap_x` horizontally and `gap_y` vertically (default: same as gap_x)."""
    gap_y = gap_x if gap_y is None else gap_y
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        result: list[list[int]] = []
        for box in boxes:
            for i, other in enumerate(result):
                if (box[0] <= other[2] + gap_x and box[2] >= other[0] - gap_x and
                        box[1] <= other[3] + gap_y and box[3] >= other[1] - gap_y):
                    result[i] = [min(box[0], other[0]), min(box[1], other[1]),
                                 max(box[2], other[2]), max(box[3], other[3])]
                    merged = True
                    break
            else:
                result.append(box)
        boxes = result
    return boxes


class Detector:
    """Turns raw frames into car bounding boxes (x, y, w, h)."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.scale = working_scale(width, height)
        self.work_size = (round(width * self.scale), round(height * self.scale))
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=True)
        self.k_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (KERNEL_SMALL, KERNEL_SMALL))
        self.k_big = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (KERNEL_BIG, KERNEL_BIG))
        self.mask = None       # last cleaned mask, at working size
        self.skipped = False   # True when the last frame was a global change

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        """Returns car boxes (x, y, w, h) in the frame's own coordinates."""
        if self.scale != 1.0:
            interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, self.work_size, interpolation=interp)
        blurred = cv2.GaussianBlur(frame, (BLUR_SIZE, BLUR_SIZE), 0)
        fg = self.back_sub.apply(blurred)
        _, fg = cv2.threshold(fg, SHADOW_THRESHOLD, 255, cv2.THRESH_BINARY)

        self.skipped = cv2.countNonZero(fg) > MAX_FOREGROUND_RATIO * fg.size
        if self.skipped:
            self.mask = fg
            return []

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_small, iterations=2)
        fg = cv2.dilate(fg, self.k_big, iterations=3)
        fg = cv2.erode(fg, self.k_big, iterations=2)
        self.mask = fg

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_BOX_W or h < MIN_BOX_H:
                continue
            if not MIN_ASPECT <= w / float(h) <= MAX_ASPECT:
                continue
            boxes.append([x, y, x + w, y + h])

        s = self.scale
        result = []
        for x1, y1, x2, y2 in merge_boxes(boxes, MERGE_GAP_X, MERGE_GAP_Y):
            x1, y1 = round(x1 / s), round(y1 / s)
            x2, y2 = min(self.width, round(x2 / s)), min(self.height, round(y2 / s))
            result.append((x1, y1, x2 - x1, y2 - y1))
        return result


# ── Tracking ─────────────────────────────────────────────────────────────────
@dataclass
class Track:
    cx: float
    cy: float
    x0: float = 0.0             # where the track started
    y0: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    unseen: int = 0
    car_id: int | None = None   # public ID, given once the track is confirmed

    def predicted(self) -> tuple[float, float]:
        steps = self.unseen + 1
        return self.cx + self.vx * steps, self.cy + self.vy * steps


class CentroidTracker:
    """Nearest-centroid tracker with motion prediction and global matching.

    - Every (detection, track) pair is ranked by distance to the track
      (predicted or last seen position, whichever is closer), and the
      closest pairs are matched first, so the result doesn't depend on the
      order of detections.
    - Confirmed cars are served before candidate tracks, so a fragment of a
      known car (e.g. the trailer it tows) is not stolen by a candidate.
    - A track becomes a car (and gets a public ID) after `min_hits`
      detections *and* once it has travelled `min_move` px: foliage and
      noise flicker in place, cars cross the frame. Public IDs are 1, 2, 3...
      with no gaps.
    - A lost track is forgotten after `max_unseen` frames, or as soon as its
      predicted position leaves the frame (the car is gone), so a car leaving
      on one side can't be confused with the next one entering there.
    """

    def __init__(self, max_distance: float, max_unseen: int = MAX_UNSEEN,
                 min_hits: int = MIN_HITS, min_move: float = MIN_MOVE,
                 frame_size: tuple[int, int] | None = None):
        self.max_distance = max_distance
        self.max_unseen = max_unseen
        self.min_hits = min_hits
        self.min_move = min_move
        self.frame_size = frame_size
        self.tracks: dict[int, Track] = {}
        self._next_track = 1
        self._next_car = 1

    def _touches_border(self, box) -> bool:
        if self.frame_size is None:
            return False
        x, y, w, h = box
        fw, fh = self.frame_size
        return x <= 1 or y <= 1 or x + w >= fw - 1 or y + h >= fh - 1

    def _left_frame(self, t: Track) -> bool:
        if self.frame_size is None:
            return False
        px, py = t.predicted()
        w, h = self.frame_size
        return not (0 <= px <= w and 0 <= py <= h)

    def update(self, boxes):
        """Returns [(car_id, (x, y, w, h)), ...] for confirmed cars seen now."""
        # Track the bottom-centre of each box (where the car touches the
        # road): the wheels and shadow are always detected, while the top of
        # a dark car against dark trees comes and goes, which makes the box
        # centre jump up and down
        centers = [(x + w / 2, y + h) for x, y, w, h in boxes]

        pairs = []
        for d, (cx, cy) in enumerate(centers):
            for tid, t in self.tracks.items():
                # Closest of the predicted and the last seen position: the
                # prediction helps fast cars, the last position keeps working
                # when the motion estimate is off (car leaving, slowing down)
                px, py = t.predicted()
                dist = min(math.hypot(cx - px, cy - py),
                           math.hypot(cx - t.cx, cy - t.cy))
                if dist < self.max_distance:
                    # confirmed cars first (False < True), then by distance
                    pairs.append((t.car_id is None, dist, d, tid))
        pairs.sort()

        matched_det: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, _, d, tid in pairs:
            if d in matched_det or tid in used_tracks:
                continue
            matched_det[d] = tid
            used_tracks.add(tid)

        confirmed = []
        for d, (cx, cy) in enumerate(centers):
            tid = matched_det.get(d)
            if tid is None:
                tid = self._next_track
                self._next_track += 1
                t = self.tracks[tid] = Track(cx, cy, x0=cx, y0=cy)
                used_tracks.add(tid)
            else:
                t = self.tracks[tid]
                # A box cut by the frame edge has a shifted centre: keep the
                # previous velocity instead of learning a wrong one
                if not self._touches_border(boxes[d]):
                    steps = t.unseen + 1
                    a = VELOCITY_SMOOTHING
                    t.vx = (1 - a) * t.vx + a * (cx - t.cx) / steps
                    t.vy = (1 - a) * t.vy + a * (cy - t.cy) / steps
                t.cx, t.cy = cx, cy
                t.hits += 1
                t.unseen = 0
            if t.car_id is None and t.hits >= self.min_hits and \
                    math.hypot(t.cx - t.x0, t.cy - t.y0) >= self.min_move:
                t.car_id = self._next_car
                self._next_car += 1
            if t.car_id is not None:
                confirmed.append((t.car_id, boxes[d]))

        # Tracks not seen this frame age; they are dropped after max_unseen
        # frames or once their predicted position is out of the frame
        for tid in list(self.tracks):
            if tid in used_tracks:
                continue
            t = self.tracks[tid]
            t.unseen += 1
            if t.unseen > self.max_unseen or self._left_frame(t):
                del self.tracks[tid]
        return confirmed


# ── Per-car statistics ───────────────────────────────────────────────────────
@dataclass
class CarStats:
    first_frame: int
    first_cx: float
    last_frame: int = 0
    last_cx: float = 0.0
    best_area: int = -1
    best_frame: int = 0
    best_bbox: tuple = field(default_factory=tuple)

    def update(self, frame_index: int, bbox) -> None:
        x, y, w, h = bbox
        self.last_frame, self.last_cx = frame_index, x + w / 2
        if w * h > self.best_area:
            self.best_area, self.best_frame, self.best_bbox = w * h, frame_index, bbox


def build_report(stats: dict[int, CarStats], fps: float) -> list[dict]:
    report = []
    for car_id in sorted(stats):
        s = stats[car_id]
        dx = s.last_cx - s.first_cx
        report.append({
            "id": int(car_id),
            "closest_frame": int(s.best_frame),
            "closest_time_sec": round(s.best_frame / fps, 2),
            "bbox": [int(v) for v in s.best_bbox],
            "area": int(s.best_area),
            "first_seen_sec": round(s.first_frame / fps, 2),
            "last_seen_sec": round(s.last_frame / fps, 2),
            "direction": "right" if dx > 0 else "left" if dx < 0 else "unknown",
        })
    return report


# ── Drawing ──────────────────────────────────────────────────────────────────
def draw_car(frame, car_id: int, bbox) -> None:
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label_y = y - 10 if y >= 30 else y + 25   # keep the label inside the frame
    cv2.putText(frame, f"ID {car_id}", (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_hud(frame, frame_index: int, total: int, cars: int, proc_fps: float) -> None:
    total_txt = f"/{total}" if total > 0 else ""
    text = (f"Frame {frame_index}{total_txt}  |  Cars: {cars}  |  "
            f"Processing: {proc_fps:.0f} fps")
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)


# ── Main loop ────────────────────────────────────────────────────────────────
# Note: no "window closed?" check. On Windows, getWindowProperty(...,
# WND_PROP_VISIBLE) returns 0 after a few seconds for a 1080p window even
# though it is still open, which stopped playback on its own.

def process_video(input_path: Path, output_path: Path | None, json_path: Path | None,
                  display: bool = True, verbose: bool = True) -> list[dict]:
    """Runs the full pipeline and returns the JSON report (list of cars)."""
    log = print if verbose else (lambda *a, **k: None)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0 or fps > 1000:
        log(f"[WARN] Invalid FPS ({fps}), using {FALLBACK_FPS}")
        fps = FALLBACK_FPS
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size {width}x{height}: {input_path}")

    log(f"Video loaded  : {input_path}")
    log(f"Resolution    : {width} x {height}")
    log(f"FPS           : {fps:.2f}")
    log(f"Total frames  : {total if total > 0 else 'unknown'}")

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

    detector = Detector(width, height)
    tracker = CentroidTracker(
        max_distance=MAX_DISTANCE / detector.scale,
        max_unseen=max(1, round(MAX_UNSEEN_SEC * fps)),
        min_hits=MIN_HITS,
        min_move=MIN_MOVE / detector.scale,
        frame_size=(width, height))
    stats: dict[int, CarStats] = {}
    frame_delay = 1.0 / fps
    proc_fps = 0.0
    frame_index = 0
    skipped_frames = 0

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                log("End of video.")
                break

            boxes = detector.detect(frame)
            skipped_frames += detector.skipped
            for car_id, bbox in tracker.update(boxes):
                draw_car(frame, car_id, bbox)
                x, y, w, h = bbox
                if car_id not in stats:
                    stats[car_id] = CarStats(first_frame=frame_index,
                                             first_cx=x + w / 2)
                stats[car_id].update(frame_index, bbox)

            draw_hud(frame, frame_index, total, len(stats), proc_fps)
            if writer is not None:
                writer.write(frame)

            elapsed = max(time.perf_counter() - t0, 1e-6)
            proc_fps = 0.9 * proc_fps + 0.1 * (1 / elapsed) if proc_fps else 1 / elapsed

            if display:
                try:
                    mask = detector.mask
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = cv2.resize(mask, (width, height),
                                          interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Mask", mask)
                    cv2.imshow("Video", frame)
                except cv2.error:
                    log("[WARN] No GUI available (headless OpenCV?), "
                        "continuing without preview")
                    display = False
                else:
                    # Play at the video's real speed, minus processing time
                    wait_ms = max(1, int((frame_delay - elapsed) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), 27):
                        log("Playback stopped by user.")
                        break

            frame_index += 1
    except KeyboardInterrupt:
        log("\nInterrupted, saving what was processed so far.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    report = build_report(stats, fps)

    if skipped_frames:
        log(f"Skipped {skipped_frames} frame(s) with a global change "
            "(lighting, camera motion or background warm-up).")
    log("\n" + "=" * 72)
    log(f"CARS DETECTED: {len(report)}   (closest moment = largest bounding box)")
    log("=" * 72)
    log(f"{'ID':>4}  {'Seen (s)':>13}  {'Dir':>5}  {'Closest':>8}  {'BBox (x,y,w,h)'}")
    log("-" * 72)
    for car in report:
        seen = f"{car['first_seen_sec']:.1f}-{car['last_seen_sec']:.1f}"
        log(f"{car['id']:>4}  {seen:>13}  {car['direction']:>5}  "
            f"{car['closest_time_sec']:>7.2f}s  {tuple(car['bbox'])}")
    log("=" * 72)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        log(f"\nSaved results to {json_path}")
    if output_path is not None:
        log(f"Saved annotated video to {output_path}")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and track cars in a fixed-camera video (OpenCV).")
    p.add_argument("-i", "--input", type=Path, default=BASE_DIR / "cars.mp4",
                   help="input video (default: cars.mp4 next to app.py)")
    p.add_argument("-o", "--output", type=Path, default=BASE_DIR / "output.mp4",
                   help="annotated output video (default: output.mp4)")
    p.add_argument("-j", "--json", type=Path, default=BASE_DIR / "results.json",
                   help="JSON report (default: results.json)")
    p.add_argument("--no-video", action="store_true",
                   help="don't write the annotated video")
    p.add_argument("--no-display", action="store_true",
                   help="no preview windows, process as fast as possible")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        print(f"[ERROR] Video not found: {args.input}", file=sys.stderr)
        return 1
    if not args.no_video and args.output.resolve() == args.input.resolve():
        print("[ERROR] Output video would overwrite the input.", file=sys.stderr)
        return 1
    try:
        process_video(args.input, None if args.no_video else args.output,
                      args.json, display=not args.no_display)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
