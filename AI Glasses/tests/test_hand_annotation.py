"""Offline checks: labels, timestamps, resume, Unicode paths, and CPU inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import annotate_hands as tool


class UtilityTests(unittest.TestCase):
    def test_timestamps_are_strictly_monotonic(self):
        previous = -1
        for index, raw in enumerate((0.0, 0.0, float("nan"), -1.0, 200.2, 200.3, 0.0)):
            timestamp, origin = tool.frame_timestamp(raw, index, 30.0, previous)
            self.assertGreater(timestamp, previous)
            self.assertTrue(origin)
            previous = timestamp

    def test_source_timestamp_preferred(self):
        self.assertEqual(tool.frame_timestamp(333.2, 1, 30.0, 0), (333, "source"))

    def test_frame_count_requires_matching_exact_count(self):
        tool.check_frame_count(377, 384, 377)
        tool.check_frame_count(13, 13, None)
        with self.assertRaises(ValueError):
            tool.check_frame_count(377, 384, None)
        with self.assertRaises(ValueError):
            tool.check_frame_count(376, 384, 377)
        with self.assertRaises(ValueError):
            tool.check_frame_count(378, 384, 377)

    def test_no_hand_is_review_not_ground_truth_negative(self):
        hands, flags = tool.serialize_hands(SimpleNamespace(hand_landmarks=[], handedness=[]))
        self.assertEqual(hands, [])
        self.assertEqual(flags, ["no_detection"])

    def test_valid_hand_keeps_raw_coordinates_and_classification_score(self):
        points = [SimpleNamespace(x=0.3, y=0.4, z=-0.02) for _ in range(21)]
        category = SimpleNamespace(category_name="Left", score=0.9)
        hands, flags = tool.serialize_hands(SimpleNamespace(
            hand_landmarks=[points], handedness=[[category]]))
        self.assertEqual(len(hands[0]["landmarks_image"]), 21)
        self.assertEqual(hands[0]["landmarks_image"][0], [0.3, 0.4, -0.02])
        self.assertEqual(hands[0]["handedness_score"], 0.9)
        self.assertNotIn("confidence", hands[0])
        self.assertEqual(flags, [])

    def test_outside_image_coordinates_flagged_not_clipped(self):
        points = [SimpleNamespace(x=1.1, y=0.4, z=0.0) for _ in range(21)]
        hands, flags = tool.serialize_hands(SimpleNamespace(hand_landmarks=[points], handedness=[]))
        self.assertEqual(hands[0]["landmarks_image"][0][0], 1.1)
        self.assertIn("landmarks_outside_image", flags)

    def test_invalid_landmarks_rejected(self):
        for points in ([SimpleNamespace(x=0.2, y=0.4, z=0.0)] * 20,
                       [SimpleNamespace(x=float("nan"), y=0.4, z=0.0)] * 21):
            hands, flags = tool.serialize_hands(SimpleNamespace(hand_landmarks=[points], handedness=[]))
            self.assertFalse(hands)
            self.assertIn("invalid_landmarks", flags)

    def test_recursive_scan_excludes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            output = source / "results"
            output.mkdir()
            (source / "手部.MP4").touch()
            (source / "ignore.txt").touch()
            (output / "preview.mp4").touch()
            self.assertEqual(tool.collect_inputs(source, output), [source / "手部.MP4"])

    def test_resume_validates_labels_and_preserves_incomplete_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job, previous = tool.reserve_job(output, "test", "abcdef")
            self.assertIsNone(previous)
            labels = job / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")
            (job / "preview.jpg").write_bytes(b"fixture")
            tool.write_json(job / "summary.json", {
                "status": "complete", "signature": "abcdef", "preview_file": "preview.jpg",
                "labels_sha256": tool.file_sha256(labels), "sample_images": []})
            resumed, previous = tool.reserve_job(output, "test", "abcdef")
            self.assertEqual(resumed, job)
            self.assertIsNotNone(previous)
            labels.write_text("changed", encoding="utf-8")
            retried, previous = tool.reserve_job(output, "test", "abcdef")
            self.assertIsNone(previous)
            self.assertNotEqual(retried, job)
            self.assertEqual(labels.read_text(encoding="utf-8"), "changed")


HAS_RUNTIME = tool.MODEL_PATH.is_file() and all(
    importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "mediapipe"))


@unittest.skipUnless(HAS_RUNTIME, "Requires the local teacher model and annotation environment")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import mediapipe
        import numpy
        cls.cv, cls.mp, cls.np = cv2, mediapipe, numpy
        cls.model_hash = tool.file_sha256(tool.MODEL_PATH)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "标注结果"
        self.output.mkdir()
        self.options = argparse.Namespace(num_hands=2, save_every=5, subject_id="test_only")

    def annotate(self, source):
        return tool.annotate_one(source, self.output, tool.MODEL_PATH, self.model_hash,
                                 self.options, self.cv, self.mp, self.np)

    def read_job(self):
        return next(path for path in self.output.iterdir() if path.is_dir())

    def test_image_unicode_paths_and_repeat_run(self):
        source = self.root / "无手测试.png"
        frame = self.np.full((240, 320, 3), 255, dtype=self.np.uint8)
        success, encoded = self.cv.imencode(".png", frame)
        self.assertTrue(success)
        encoded.tofile(str(source))
        original_hash = tool.file_sha256(source)
        summary, skipped = self.annotate(source)
        self.assertFalse(skipped)
        self.assertEqual(summary["processed_frames"], 1)
        self.assertEqual(summary["frames_without_detection"], 1)
        job = self.read_job()
        row = json.loads((job / "labels.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["quality_flags"], ["no_detection"])
        self.assertTrue(row["needs_review"])
        self.assertIsNone(row["gesture_label"])
        self.assertTrue((job / row["image_path"]).is_file())
        self.assertEqual(tool.file_sha256(source), original_hash)
        original_stat = (job / "labels.jsonl").stat()
        repeated, skipped = self.annotate(source)
        self.assertTrue(skipped)
        self.assertEqual(summary, repeated)
        self.assertEqual((job / "labels.jsonl").stat().st_mtime_ns, original_stat.st_mtime_ns)

    def test_video_frame_count_timestamps_sampling_and_playable_preview(self):
        source = self.root / "无手视频.avi"
        writer = self.cv.VideoWriter(str(source), self.cv.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(13):
            writer.write(self.np.full((240, 320, 3), 255, dtype=self.np.uint8))
        writer.release()
        original_hash = tool.file_sha256(source)
        summary, skipped = self.annotate(source)
        self.assertFalse(skipped)
        self.assertEqual(summary["processed_frames"], 13)
        self.assertEqual(summary["frames_without_detection"], 13)
        self.assertEqual(summary["saved_images"], 3)
        job = self.read_job()
        rows = [json.loads(line) for line in (job / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["frame_index"] for row in rows], list(range(13)))
        self.assertTrue(all(a["timestamp_ms"] < b["timestamp_ms"] for a, b in zip(rows, rows[1:])))
        self.assertEqual([row["frame_index"] for row in rows if row["image_path"]], [0, 5, 10])
        reader = self.cv.VideoCapture(str(job / summary["preview_file"]))
        decoded = 0
        while reader.read()[0]:
            decoded += 1
        reader.release()
        self.assertEqual(decoded, 13)
        self.assertEqual(tool.file_sha256(source), original_hash)

    def test_corrupt_input_never_publishes_completion(self):
        source = self.root / "bad.mp4"
        source.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            self.annotate(source)
        job = self.read_job()
        self.assertFalse((job / "summary.json").exists())
        self.assertTrue((job / "error.json").is_file())
        self.assertEqual(source.read_bytes(), b"not a video")

    def test_bundled_public_frame_has_21_point_hand(self):
        fixture = ROOT / "tests" / "fixtures" / "public_hand_frame.jpg"
        summary, skipped = self.annotate(fixture)
        self.assertFalse(skipped)
        self.assertEqual(summary["frames_with_hands"], 1)
        job = self.read_job()
        manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_relative_path"], "tests/fixtures/public_hand_frame.jpg")
        self.assertEqual(manifest["model_relative_path"], "models/teacher/hand_landmarker.task")
        row = json.loads((job / "labels.jsonl").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(row["hands"]), 1)
        for hand in row["hands"]:
            self.assertEqual(len(hand["landmarks_image"]), 21)
            self.assertTrue(all(len(point) == 3 for point in hand["landmarks_image"]))


if __name__ == "__main__":
    unittest.main()
