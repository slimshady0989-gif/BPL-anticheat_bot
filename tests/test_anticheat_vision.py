import asyncio
import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anticheat_bot as app  # noqa: E402
import anticheat_report as report_tools  # noqa: E402
import anticheat_vision as vision  # noqa: E402


class OctantSnappingTests(unittest.TestCase):
    def test_wasd_vectors_snap_to_eight_directions(self):
        vectors = []
        for step in range(4):
            angle = math.radians(step * 45)
            vectors += [(5 * math.cos(angle), -5 * math.sin(angle))] * 6
        result = vision.analyze_octant_snapping(vectors, min_speed=1.0)
        self.assertGreaterEqual(result.snapped_ratio, 0.99)
        self.assertGreaterEqual(result.distinct_directions, 3)
        self.assertGreater(result.diagonal_share, 0.2)

    def test_analog_stick_headings_do_not_snap(self):
        rng = np.random.default_rng(7)
        vectors = [
            (5 * math.cos(a), -5 * math.sin(a))
            for a in rng.uniform(0, 2 * math.pi, 400)
        ]
        result = vision.analyze_octant_snapping(vectors, min_speed=1.0)
        self.assertLess(result.snapped_ratio, 0.35)

    def test_slow_vectors_are_ignored(self):
        result = vision.analyze_octant_snapping([(0.1, 0.0)] * 10, min_speed=1.0)
        self.assertEqual(result.samples, 0)

    def test_micro_jitter_detects_alternating_residual(self):
        vectors = [(4.0 if i % 2 else -4.0, 0.0) for i in range(120)]
        self.assertGreater(vision.micro_jitter_hz(vectors, 2.0), 10.0)

    def test_smooth_motion_has_no_jitter(self):
        vectors = [(8.0, 0.0)] * 120
        self.assertEqual(vision.micro_jitter_hz(vectors, 2.0), 0.0)

    def test_strict_wasd_jitter_is_ignored_inside_impulse_window(self):
        samples = [
            vision.Sample(index / 20, index, 0, 0, 4 if index % 2 else -4, 0)
            for index in range(40)
        ]
        track = SimpleNamespace(samples=samples, entity_id="E1")
        strict_snap = vision.SnapResult(40, 0.95, 4, 0.5)
        with (
            patch.object(
                vision, "analyze_octant_snapping", return_value=strict_snap
            ),
            patch.object(vision, "micro_jitter_hz", return_value=12.0),
        ):
            finding, snap = vision.analyze_locomotion(
                track, 20, impulse_windows=[(0.0, 2.0)]
            )

        self.assertIsNone(finding)
        self.assertIsNone(snap)


class ParsingTests(unittest.TestCase):
    def test_timer_formats(self):
        self.assertEqual(vision.parse_timer_text("0:42"), 42)
        self.assertEqual(vision.parse_timer_text("round 1:05"), 65)
        self.assertEqual(vision.parse_timer_text("42"), 42)
        self.assertIsNone(vision.parse_timer_text("0:99"))
        self.assertIsNone(vision.parse_timer_text("abc"))
        self.assertIsNone(vision.parse_timer_text(""))

    def test_nickname_matching_tolerates_ocr_noise(self):
        roster = ["ShadowWolf", "Kowalski"]
        self.assertEqual(vision.match_nickname("ShadowW0lf", roster), "ShadowWolf")
        self.assertIsNone(vision.match_nickname("Unknown", roster))

    def test_killfeed_pairs(self):
        events = vision.parse_killfeed(
            "ShadowWolf  Kowalski", 12.5, roster=["ShadowWolf", "Kowalski"]
        )
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].killer, events[0].victim), ("ShadowWolf", "Kowalski"))

    def test_format_clock(self):
        self.assertEqual(vision.format_clock(65), "1:05")
        self.assertIsNone(vision.format_clock(None))

    def test_timer_tracker_rejects_outliers_and_extrapolates(self):
        tracker = vision.TimerTracker()
        self.assertTrue(tracker.add(0.0, 60))
        self.assertTrue(tracker.add(1.0, 59))
        self.assertFalse(tracker.add(2.0, 10))
        self.assertAlmostEqual(tracker.clock_at(3.0), 57.0)
        self.assertIsNone(vision.TimerTracker().clock_at(1.0))


class HudReaderTests(unittest.TestCase):
    def test_reader_uses_backend_for_timer_and_killfeed(self):
        backend = MagicMock(available=True)
        backend.read.side_effect = lambda image, kind: {
            "timer": "1:30",
            "killfeed": "Alpha  Bravo",
        }.get(kind, "")
        reader = vision.HudReader(backend, roster=["Alpha", "Bravo"])
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertTrue(reader.available)
        self.assertEqual(reader.read_timer(frame), 90)
        kills = reader.read_killfeed(frame, 3.0)
        self.assertEqual((kills[0].killer, kills[0].victim), ("Alpha", "Bravo"))

    def test_null_backend_is_unavailable_and_returns_nothing(self):
        reader = vision.HudReader(vision.NullOcrBackend())
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertFalse(reader.available)
        self.assertIsNone(reader.read_timer(frame))


class YoloDecodeTests(unittest.TestCase):
    def test_decode_returns_person_box_in_frame_coordinates(self):
        output = np.zeros((1, 84, 8400), np.float32)
        output[0, :4, 10] = [320, 320, 100, 200]
        output[0, 4, 10] = 0.9
        detections = vision.decode_yolo_output(
            output, ratio=0.5, pad=(0, 140), frame_size=(1280, 720),
            confidence=0.25, nms_iou=0.45,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual((detections[0].x, detections[0].w, detections[0].h), (540, 200, 400))

    def test_decode_returns_nothing_below_confidence(self):
        output = np.zeros((1, 84, 8400), np.float32)
        output[0, 4, 5] = 0.05
        self.assertEqual(
            vision.decode_yolo_output(output, 1.0, (0, 0), (640, 640), 0.25, 0.45), []
        )


class ProfileThresholdTests(unittest.TestCase):
    def test_few_samples_use_fallback(self):
        value, source = report_tools.personalized_threshold({"sample_count": 2}, 26.5)
        self.assertEqual(value, 26.5)
        self.assertEqual(source, "control-profile")
        self.assertEqual(report_tools.personalized_threshold(None, 30.0)[0], 30.0)

    def test_three_sigma_threshold_from_history(self):
        n = 60
        profile = {
            "sample_count": n,
            "speed_mean": 10.0,
            "speed_m2": (n - 1) * 2.0 ** 2,
            "lag_clip_count": 0,
        }
        value, source = report_tools.personalized_threshold(profile, 26.5)
        self.assertAlmostEqual(value, 16.0, places=3)
        self.assertTrue(source.startswith("player-profile"))

    def test_lag_prone_player_gets_wider_threshold_and_longer_run(self):
        base = {"sample_count": 60, "speed_mean": 10.0, "speed_m2": 59 * 4.0}
        lagged = dict(base, lag_clip_count=30)
        clean = dict(base, lag_clip_count=0)
        self.assertGreater(
            report_tools.personalized_threshold(lagged, 26.5)[0],
            report_tools.personalized_threshold(clean, 26.5)[0],
        )
        self.assertGreater(
            report_tools.adaptive_min_run(lagged, 5), report_tools.adaptive_min_run(clean, 5)
        )

    def test_threshold_and_probability_are_versioned_to_kolejka_one_baseline(self):
        baseline = {
            "league_round": 1,
            "calibration_version": "k1-2026-v1",
            "sample_count": 30,
            "base_avg_speed": 10.0,
            "speed_m2": 29 * 4.0,
        }
        threshold, source = report_tools.personalized_threshold(
            None, 99.0, calibration_baseline=baseline
        )
        z_score, probability = report_tools.calibration_statistics(baseline, 16.0)

        self.assertEqual(threshold, 16.0)
        self.assertIn("k1-2026-v1", source)
        self.assertEqual(z_score, 3.0)
        self.assertAlmostEqual(probability, 0.0013499, places=6)


def _finding(kind="HARDWARE_EMULATOR", offender="ShadowWolf"):
    return vision.Finding(
        kind=kind,
        label=vision.LABEL_HARDWARE,
        entity_id="E1",
        video_second=12.4,
        evidence="snap 97%",
        confidence=0.8,
        clock="1:48",
        offender=offender,
    )


class ReportTests(unittest.TestCase):
    def _result(self, with_finding=True):
        multi = vision.MultiPlayerResult(
            findings=[_finding()] if with_finding else [],
            matrix_rows=[(12.4, 744, "E1", 640.0, 360.0, 5.0, -5.0)],
            detector_name="yolov8n",
        )
        return {
            "alert": with_finding,
            "raport": "verdict",
            "multi": multi,
            "anomaly_second": 4.0,
            "anomaly_clock": "1:56",
        }

    def test_summarize_prefers_multi_player_finding(self):
        offender, kind, second, clock = report_tools.summarize_anomaly(self._result())
        self.assertEqual(offender, "ShadowWolf")
        self.assertEqual(kind, vision.LABEL_HARDWARE)
        self.assertEqual((second, clock), (12.4, "1:48"))

    def test_summarize_clean_result(self):
        self.assertEqual(
            report_tools.summarize_anomaly(self._result(False)), (None, None, None, None)
        )

    def test_detailed_log_contains_required_sections(self):
        text = report_tools.build_detailed_analysis(
            self._result(), {"filename": "clip.mp4", "player_name": "Victim", "player_id": 1}
        )
        self.assertIn(report_tools.REPORT_ID_TOKEN, text)
        self.assertIn("clip.mp4", text)
        self.assertIn("ShadowWolf", text)
        self.assertIn("Victim", text)

    def test_attachment_has_report_id_in_name_and_body(self):
        report = {
            "id": 77,
            "detail_log": f"REPORT {report_tools.REPORT_ID_TOKEN}\nbody",
        }
        attachment = app._detail_attachment(report)
        self.assertEqual(attachment.filename, "Detailed_Analysis_77.txt")
        body = attachment.fp.read().decode("utf-8")
        self.assertIn("REPORT 77", body)
        self.assertNotIn(report_tools.REPORT_ID_TOKEN, body)


class EmbedFieldTests(unittest.TestCase):
    def test_embed_contains_required_fields(self):
        report = {
            "id": 5,
            "source_message_id": 1,
            "guild_id": 2,
            "player_id": 3,
            "player_name": "Victim",
            "video_url": "https://x/clip.mp4",
            "control_type": "PAD",
            "avg_speed": 3.0,
            "magnitude_variance": 1.0,
            "jitter_frequency": 2.0,
            "is_anomalous": True,
            "report_status": "PENDING",
            "raw_report_text": "txt",
            "offender_name": "ShadowWolf",
            "anomaly_type": vision.LABEL_HARDWARE,
            "anomaly_second": 12.4,
            "anomaly_clock": "1:48",
        }
        embed = app._report_embed(report)
        names = [field.name for field in embed.fields]
        self.assertIn("ID raportu w bazie Neon", names)
        for expected in (
            "Wykryty Przeciwnik",
            "Typ Wykrytej Anomalii",
            "Czas zdarzenia na wideo",
            "1:48",
            "ShadowWolf",
            "12.40s",
        ):
            self.assertIn(expected, embed.description)


class PhysicsConstantsTests(unittest.TestCase):
    def test_labels_match_specification(self):
        self.assertIn("XIM", vision.LABEL_HARDWARE)
        self.assertIn("TELEPORT", vision.LABEL_TELEPORT)
        self.assertIn("AIR-STRAFE", vision.LABEL_AIRSTRAFE)

    def test_impulse_detector_exposes_windows_api(self):
        self.assertTrue(hasattr(vision.ImpulseWindowDetector, "__init__"))
        self.assertTrue(hasattr(vision.PhysicsValidator, "__init__"))


class AudioExtractionTests(unittest.TestCase):
    def test_network_and_local_sources_use_different_timeouts(self):
        completed = vision.subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout=b"\x00\x00" * 16000
        )
        with (
            patch.object(vision.shutil, "which", return_value="ffmpeg"),
            patch.object(vision.subprocess, "run", return_value=completed) as run,
        ):
            vision.extract_audio_envelope("https://example.test/live", 30, 30)
            self.assertEqual(run.call_args.kwargs["timeout"], 60)
            vision.extract_audio_envelope("clip.mp4", 30, 30)
            self.assertEqual(run.call_args.kwargs["timeout"], 300)

    def test_ffmpeg_failures_and_empty_payload_return_none(self):
        failures = (
            vision.subprocess.TimeoutExpired("ffmpeg", 60),
            vision.subprocess.CalledProcessError(1, "ffmpeg"),
            OSError("ffmpeg is unavailable"),
            RuntimeError("unexpected failure"),
        )
        with patch.object(vision.shutil, "which", return_value="ffmpeg"):
            for failure in failures:
                with self.subTest(failure=type(failure).__name__):
                    with patch.object(
                        vision.subprocess, "run", side_effect=failure
                    ):
                        self.assertIsNone(
                            vision.extract_audio_envelope(
                                "https://example.test/live", 30, 30
                            )
                        )

            empty = vision.subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=0, stdout=b""
            )
            with patch.object(vision.subprocess, "run", return_value=empty):
                with self.assertLogs(vision.LOGGER, level="WARNING"):
                    self.assertIsNone(
                        vision.extract_audio_envelope("clip.mp4", 30, 30)
                    )


class ClipSmokeTests(unittest.TestCase):
    CLIP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test clip")

    def test_reference_clips_open_and_have_expected_layout(self):
        import cv2

        if not os.path.isdir(self.CLIP_DIR):
            self.skipTest("reference clips not present")
        clips = [f for f in os.listdir(self.CLIP_DIR) if f.lower().endswith(".mp4")]
        if not clips:
            self.skipTest("reference clips not present")
        for name in clips:
            capture = cv2.VideoCapture(os.path.join(self.CLIP_DIR, name))
            try:
                ok, frame = capture.read()
                self.assertTrue(ok, name)
                self.assertEqual(frame.shape[:2], (720, 1280), name)
                reader = vision.HudReader(vision.NullOcrBackend())
                self.assertIsNone(reader.read_timer(frame))
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
