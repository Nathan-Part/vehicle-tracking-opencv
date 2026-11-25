import cv2
import sys
import math
import json

# ── Configuration ──────────────────────────────────────────────
VIDEO_PATH = r"c:\Users\natou\Desktop\ITC-2\cars.mp4"

# ── Open the video file ───────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

# ── Read and print video metadata ─────────────────────────────
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded  : {VIDEO_PATH}")
print(f"Resolution    : {width} x {height}")
print(f"FPS           : {fps:.2f}")
print(f"Total frames  : {total_frames}")

# ── Background subtractor ─────────────────────────────────────
# detectShadows=True → shadows become gray (~127) so we can filter them out
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=700, varThreshold=120, detectShadows=True)

# ── Detection tuning ──────────────────────────────────────────
MIN_CONTOUR_AREA = 6000   # ignore contours smaller than this (pixels)

# ── Step 4: Tracking setup ────────────────────────────────────
# Each tracked object: {id: (cx, cy, frames_unseen, frames_seen)}
tracked_objects = {}    # objects from the PREVIOUS frame
next_id = 1             # counter for assigning new IDs
MAX_DISTANCE = 350      # max pixels between frames to consider it the same car
MAX_UNSEEN = 8          # keep an object in memory for this many frames if not detected
MIN_HITS = 3            # object must be seen this many frames before we draw it

# ── Step 5: Closest moment tracking ──────────────────────────
# For each ID, store its biggest bounding box (= closest to camera)
closest_moment = {}

# ── Calculate delay between frames (milliseconds) ─────────────
delay = int(1000 / fps) if fps > 0 else 30   # fallback to ~33 ms

# ── Step 6: Video writer (saves annotated video to output.mp4) ──
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = r"c:\Users\natou\Desktop\ITC-2\output.mp4"
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# ── Playback loop ─────────────────────────────────────────────
frame_index = 0

while True:
    ret, frame = cap.read()          # read the next frame
    if not ret:
        print("End of video (or read error).")
        break

    # ── 1) Blur the frame to reduce camera noise BEFORE detection ──
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # ── 2) Background subtraction ──────────────────────────────
    fg_mask = back_sub.apply(blurred)

    # ── 3) Remove shadows (gray pixels) → keep only pure white ─
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # ── 4) Morphological cleanup ───────────────────────────────
    # Small kernel to remove tiny noise dots
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Big kernel to merge fragments of the same car into one blob
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # OPEN with small kernel → removes small noise dots
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Dilate with big kernel → bridges gaps between fragments
    fg_mask = cv2.dilate(fg_mask, kernel_big, iterations=3)
    # Erode back so the box isn't too bloated
    fg_mask = cv2.erode(fg_mask, kernel_big, iterations=2)

    # ── 5) Find contours and collect valid detections ───────────
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Build a list of valid bounding boxes: [(x, y, w, h), ...]
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter: must be wide and tall enough to be a car
        if w < 40 or h < 130:
            continue
        aspect = w / float(h)
        if aspect < 0.3 or aspect > 4.0:
            continue
        boxes.append([x, y, x + w, y + h])   # store as [x1, y1, x2, y2]

    # ── Merge overlapping or nearby boxes (so one car = one box) ─
    MERGE_GAP = 100  # merge boxes within 100px of each other
    merged = True
    while merged:
        merged = False
        new_boxes = []
        while boxes:
            box = boxes.pop(0)
            found = False
            for i, other in enumerate(new_boxes):
                # Check if the two boxes overlap OR are close
                if box[0] <= other[2] + MERGE_GAP and box[2] >= other[0] - MERGE_GAP and \
                   box[1] <= other[3] + MERGE_GAP and box[3] >= other[1] - MERGE_GAP:
                    # Merge: take the bounding box of both
                    new_boxes[i] = [min(box[0], other[0]), min(box[1], other[1]),
                                    max(box[2], other[2]), max(box[3], other[3])]
                    found = True
                    merged = True
                    break
            if not found:
                new_boxes.append(box)
        boxes = new_boxes

    # Convert merged boxes to detections: [(cx, cy, x, y, w, h), ...]
    detections = []
    for (x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2
        detections.append((cx, cy, x1, y1, w, h))

    # ── 6) Simple tracking: match detections to previous objects ─
    new_tracked = {}           # will replace tracked_objects at end of frame
    used_ids = set()           # IDs already matched this frame

    for (cx, cy, x, y, w, h) in detections:
        best_id = None
        best_dist = MAX_DISTANCE  # only match if closer than this

        # Compare this detection's center to every known object
        for obj_id, (ox, oy, _, seen) in tracked_objects.items():
            dist = math.hypot(cx - ox, cy - oy)  # Euclidean distance
            if dist < best_dist and obj_id not in used_ids:
                best_dist = dist
                best_id = obj_id

        # Count how many times this object has been seen
        if best_id is not None:
            prev_seen = tracked_objects[best_id][3]
        else:
            prev_seen = 0
            best_id = next_id
            next_id += 1

        hits = prev_seen + 1
        used_ids.add(best_id)
        new_tracked[best_id] = (cx, cy, 0, hits)

        # Only draw if object has been seen for MIN_HITS frames (filters flicker)
        if hits >= MIN_HITS:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {best_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ── Step 5: Track closest moment (biggest box = closest) ─
            area = w * h
            if best_id not in closest_moment or area > closest_moment[best_id]["area"]:
                closest_moment[best_id] = {
                    "area": area,
                    "frame": frame_index,
                    "bbox": (x, y, w, h)
                }

    # Keep unmatched objects in memory for a few frames (grace period)
    for obj_id, (ox, oy, unseen, seen) in tracked_objects.items():
        if obj_id not in used_ids and unseen < MAX_UNSEEN:
            new_tracked[obj_id] = (ox, oy, unseen + 1, seen)

    # Update tracked objects for the next frame
    tracked_objects = new_tracked

    # Overlay frame counter + FPS on the frame
    text = f"Frame: {frame_index}  |  FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Mask", fg_mask)       # debug: see the cleaned mask
    cv2.imshow("Video", frame)       # display the frame with boxes

    # Step 6: Write annotated frame to output video
    writer.write(frame)

    frame_index += 1

    # Press 'q' to quit early
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Playback stopped by user.")
        break

# ── Cleanup ───────────────────────────────────────────────────
cap.release()
writer.release()
cv2.destroyAllWindows()

# ── Step 5: Print summary of closest moments ─────────────────
print("\n" + "=" * 60)
print("CLOSEST MOMENT PER CAR (largest bounding box)")
print("=" * 60)
print(f"{'ID':>5}  {'Frame':>6}  {'Time (s)':>9}  {'BBox (x,y,w,h)'}")
print("-" * 60)
for car_id in sorted(closest_moment.keys()):
    info = closest_moment[car_id]
    f = info["frame"]
    t = f / fps if fps > 0 else 0
    bx, by, bw, bh = info["bbox"]
    print(f"{car_id:>5}  {f:>6}  {t:>8.2f}s  ({bx}, {by}, {bw}, {bh})")
print("=" * 60)

# ── Step 6: Export results to JSON ───────────────────────────
results = []
for car_id in sorted(closest_moment.keys()):
    info = closest_moment[car_id]
    f = info["frame"]
    bx, by, bw, bh = info["bbox"]
    results.append({
        "id": int(car_id),
        "closest_frame": int(f),
        "closest_time_sec": round(f / fps, 2) if fps > 0 else 0,
        "bbox": [int(bx), int(by), int(bw), int(bh)],
        "area": int(info["area"])
    })

json_path = r"c:\Users\natou\Desktop\ITC-2\results.json"
with open(json_path, "w") as jf:
    json.dump(results, jf, indent=2)

print(f"\nSaved results to {json_path}")
print(f"Saved annotated video to {out_path}")
