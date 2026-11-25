"""Tests for app.py. Run with:  python -m unittest -v"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


class MergeBoxesTest(unittest.TestCase):
    def test_overlapping_boxes_are_merged(self):
        self.assertEqual(app.merge_boxes([[0, 0, 10, 10], [5, 5, 20, 20]], 0),
                         [[0, 0, 20, 20]])

    def test_close_boxes_are_merged(self):
        self.assertEqual(app.merge_boxes([[0, 0, 10, 10], [50, 0, 60, 10]], 100),
                         [[0, 0, 60, 10]])

    def test_far_boxes_are_kept_apart(self):
        boxes = [[0, 0, 10, 10], [500, 0, 510, 10]]
        self.assertEqual(app.merge_boxes(boxes, 100), boxes)

    def test_chain_is_merged_into_one_box(self):
        boxes = [[0, 0, 10, 10], [400, 0, 410, 10], [200, 0, 210, 10]]
        self.assertEqual(app.merge_boxes(boxes, 200), [[0, 0, 410, 10]])

    def test_input_is_not_modified(self):
        boxes = [[0, 0, 10, 10], [5, 5, 20, 20]]
        app.merge_boxes(boxes, 0)
        self.assertEqual(boxes, [[0, 0, 10, 10], [5, 5, 20, 20]])


class WorkingScaleTest(unittest.TestCase):
    def test_reference_size_is_unchanged(self):
        self.assertEqual(app.working_scale(1920, 1080), 1.0)

    def test_smaller_and_portrait_videos(self):
        self.assertAlmostEqual(app.working_scale(1280, 720), 1.5)
        self.assertAlmostEqual(app.working_scale(720, 1280), 1.5)


def box_at(cx, cy, w=100, h=60):
    return (int(cx - w / 2), int(cy - h / 2), w, h)


class TrackerTest(unittest.TestCase):
    def test_car_is_confirmed_after_min_hits(self):
        tr = app.CentroidTracker(max_distance=100, min_hits=3)
        self.assertEqual(tr.update([box_at(100, 100)]), [])
        self.assertEqual(tr.update([box_at(110, 100)]), [])
        self.assertEqual([c for c, _ in tr.update([box_at(120, 100)])], [1])

    def test_short_gap_keeps_the_id(self):
        tr = app.CentroidTracker(max_distance=100, max_unseen=8, min_hits=1)
        tr.update([box_at(100, 100)])
        for _ in range(5):
            tr.update([])
        self.assertEqual([c for c, _ in tr.update([box_at(110, 100)])], [1])

    def test_long_gap_gives_a_new_id(self):
        tr = app.CentroidTracker(max_distance=100, max_unseen=3, min_hits=1)
        tr.update([box_at(100, 100)])
        for _ in range(4):
            tr.update([])
        self.assertEqual([c for c, _ in tr.update([box_at(100, 100)])], [2])

    def test_flicker_does_not_create_id_gaps(self):
        tr = app.CentroidTracker(max_distance=50, min_hits=3)
        tr.update([box_at(900, 900)])          # one-frame noise, never confirmed
        for x in (100, 110, 120):
            out = tr.update([box_at(x, 100)])
        self.assertEqual([c for c, _ in out], [1])

    def test_matching_does_not_depend_on_detection_order(self):
        def run(order):
            tr = app.CentroidTracker(max_distance=300, min_hits=1)
            tr.update([box_at(100, 100), box_at(300, 100)])
            # Each car moved 20 px; the first detection is closer to the
            # *other* car's centre in a naive first-come matcher.
            frame = [box_at(280, 100), box_at(120, 100)]
            out = tr.update(frame if order else frame[::-1])
            return {cid: b[0] + b[2] // 2 for cid, b in out}
        self.assertEqual(run(True), run(False))
        self.assertEqual(run(True), {1: 120, 2: 280})

    def test_prediction_follows_a_fast_car(self):
        # 150 px per frame with max_distance 100: only the motion prediction
        # can keep the same ID.
        tr = app.CentroidTracker(max_distance=100, min_hits=1)
        tr.update([box_at(100, 100)])
        tr.update([box_at(190, 100)])           # within 100 px, gives velocity
        ids = [c for c, _ in tr.update([box_at(290, 100)])]
        ids += [c for c, _ in tr.update([box_at(390, 100)])]
        self.assertEqual(ids, [1, 1])


class ReportTest(unittest.TestCase):
    def test_report_fields(self):
        s = app.CarStats(first_frame=30, first_cx=1000)
        s.update(30, (950, 10, 100, 50))
        s.update(60, (500, 10, 300, 200))
        s.update(90, (100, 10, 100, 50))
        [car] = app.build_report({1: s}, fps=30)
        self.assertEqual(car["closest_frame"], 60)
        self.assertEqual(car["closest_time_sec"], 2.0)
        self.assertEqual(car["bbox"], [500, 10, 300, 200])
        self.assertEqual(car["area"], 60000)
        self.assertEqual(car["first_seen_sec"], 1.0)
        self.assertEqual(car["last_seen_sec"], 3.0)
        self.assertEqual(car["direction"], "left")


def make_video(path, frames=90, size=(320, 240), flash_at=()):
    """Gray road, one car going right in the top lane, one going left below."""
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    rng = np.random.default_rng(0)
    for i in range(frames):
        f = np.full((h, w, 3), 100, np.uint8)
        f = cv2.add(f, rng.integers(0, 6, f.shape, dtype=np.uint8))
        x1 = -80 + i * 6
        x2 = w + 10 - i * 6
        cv2.rectangle(f, (x1, 40), (x1 + 70, 90), (30, 30, 200), -1)
        cv2.rectangle(f, (x2, 150), (x2 + 70, 200), (200, 60, 30), -1)
        if i in flash_at:
            f = cv2.convertScaleAbs(f, alpha=1.0, beta=120)
        out.write(f)
    out.release()


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_cars_opposite_directions(self):
        video = self.dir / "two.mp4"
        make_video(video)
        out, js = self.dir / "out" / "o.mp4", self.dir / "r.json"
        report = app.process_video(video, out, js, display=False, verbose=False)
        self.assertEqual(len(report), 2)
        self.assertEqual(sorted(c["direction"] for c in report), ["left", "right"])
        self.assertEqual(json.loads(js.read_text(encoding="utf-8")), report)
        self.assertTrue(out.is_file() and out.stat().st_size > 0)

    def test_light_flash_does_not_create_cars(self):
        video = self.dir / "flash.mp4"
        make_video(video, flash_at=(40, 41))
        report = app.process_video(video, None, None, display=False, verbose=False)
        self.assertEqual(len(report), 2)

    def test_unreadable_file_raises(self):
        bad = self.dir / "bad.mp4"
        bad.write_text("not a video")
        with self.assertRaises(RuntimeError):
            app.process_video(bad, None, None, display=False, verbose=False)

    def test_cli_missing_file_returns_1(self):
        self.assertEqual(app.main(["-i", str(self.dir / "nope.mp4"),
                                   "--no-display"]), 1)

    def test_cli_refuses_to_overwrite_input(self):
        video = self.dir / "in.mp4"
        make_video(video, frames=5)
        self.assertEqual(app.main(["-i", str(video), "-o", str(video),
                                   "--no-display"]), 1)


if __name__ == "__main__":
    unittest.main()
