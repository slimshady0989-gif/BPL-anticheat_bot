import asyncio
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import cv2
import discord
import numpy as np

import anticheat_bot as app


class FakeStream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.read_sizes = []

    async def read(self, size):
        self.read_sizes.append(size)
        return next(self.chunks, b"")


class FakeResponse:
    status = 200
    content_length = None

    def __init__(self, chunks):
        self.content = FakeStream(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def get(self, url):
        return self.response


class FakeChannel:
    def __init__(self, channel_id, name, overwrites=None):
        self.id = channel_id
        self.name = name
        self.overwrites = overwrites or {}
        self.edit = AsyncMock()
        self.set_permissions = AsyncMock(side_effect=self._set_permissions)

    def overwrites_for(self, target):
        return self.overwrites.get(target, discord.PermissionOverwrite())

    async def _set_permissions(self, target, overwrite=None):
        if overwrite is None:
            self.overwrites.pop(target, None)
        else:
            self.overwrites[target] = overwrite


class FakeCategory:
    def __init__(self, category_id, name, text_channels=None, overwrites=None):
        self.id = category_id
        self.name = name
        self.text_channels = text_channels or []
        self.overwrites = overwrites or {}


class FakeGuild:
    def __init__(self, guild_id, categories=None, bot_member=None):
        self.id = guild_id
        self.categories = categories or []
        self.me = bot_member or object()
        self.default_role = object()
        self.created_channels = []
        self.create_category = AsyncMock()
        self.create_text_channel = AsyncMock(side_effect=self._create_text_channel)

    async def _create_text_channel(self, name, category):
        channel = FakeChannel(500 + len(self.created_channels), name)
        category.text_channels.append(channel)
        self.created_channels.append(channel)
        return channel


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeDatabasePool:
    def __init__(self, profiles=None, report=None):
        self.profiles = profiles or {
            "KBM": {
                "control_type": "KBM",
                "base_avg_speed": 14.5,
                "max_legal_velocity": 38.0,
                "sample_count": 50,
                "speed_m2": 50 * ((38.0 - 14.5) / 3.0) ** 2,
            },
            "PAD": {
                "control_type": "PAD",
                "base_avg_speed": 11.2,
                "max_legal_velocity": 26.5,
                "sample_count": 50,
                "speed_m2": 50 * ((26.5 - 11.2) / 3.0) ** 2,
            },
        }
        self.report = report
        self.connection = self
        self.next_report_id = 1
        self.pending_rows = []

    def acquire(self):
        return FakeAcquire(self)

    def transaction(self):
        return FakeTransaction()

    async def fetch(self, query):
        if "FROM reports" in query:
            return self.pending_rows
        if "FROM player_profiles" in query or "FROM gadgets_mechanics" in query:
            return []
        return list(self.profiles.values())

    async def fetchval(self, query, *args):
        self.report = {
            "id": self.next_report_id,
            "message_id": None,
            "source_message_id": args[0],
            "guild_id": args[1],
            "player_id": args[2],
            "player_name": args[3],
            "video_url": args[4],
            "control_type": args[5],
            "avg_speed": args[6],
            "magnitude_variance": args[7],
            "jitter_frequency": args[8],
            "is_anomalous": args[9],
            "report_status": "PENDING",
            "raw_report_text": args[10],
        }
        if len(args) >= 19:
            self.report.update(
                offender_name=args[11],
                anomaly_type=args[12],
                anomaly_second=args[13],
                anomaly_clock=args[14],
                detail_log=args[15],
                lag_detected=args[16],
                offender_avg_speed=args[17],
                offender_jitter_hz=args[18],
            )
        self.next_report_id += 1
        return self.report["id"]

    async def fetchrow(self, query, *args):
        if "FROM control_profiles" in query:
            return self.profiles[args[0]]
        if "FROM player_profiles" in query:
            return None
        return self.report

    async def execute(self, query, *args):
        if "UPDATE control_profiles" in query:
            profile = self.profiles[args[0]]
            profile["base_avg_speed"] = args[1]
            profile["sample_count"] = args[2]
            profile["speed_m2"] = args[3]
        elif "VERIFIED_CLEAN" in query:
            self.report["report_status"] = "VERIFIED_CLEAN"
            self.report["control_type"] = args[1]
        elif "VERIFIED_CHEAT" in query:
            self.report["report_status"] = "VERIFIED_CHEAT"
        elif "MANUAL_REVIEW" in query:
            self.report["report_status"] = "MANUAL_REVIEW"
        elif "SET message_id" in query:
            self.report["message_id"] = args[1]


class DatabaseInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialization_creates_tables_and_seeds_profiles_and_mechanics(self):
        pool = SimpleNamespace(execute=AsyncMock(), executemany=AsyncMock())

        await app.initialize_database(pool)

        ddl = pool.execute.await_args_list[0].args[0]
        self.assertIn("CREATE TABLE IF NOT EXISTS gadgets_mechanics", ddl)
        self.assertIn("CREATE TABLE IF NOT EXISTS control_profiles", ddl)
        self.assertIn("CREATE TABLE IF NOT EXISTS reports", ddl)
        self.assertEqual(
            pool.executemany.await_args.args[1],
            [
                ("Tactical_Sprint", 1.25, 3.0),
                ("Impulse_Grenade", 3.0, 1.8),
                ("Sliding", 1.4, 1.2),
            ],
        )
        self.assertIn("CREATE TABLE IF NOT EXISTS player_profiles", ddl)
        seeded_profiles = [
            call.args
            for call in pool.execute.await_args_list
            if "INSERT INTO control_profiles" in call.args[0]
        ]
        self.assertEqual(
            [(args[1], args[2], args[3], args[4]) for args in seeded_profiles],
            [
                ("KBM", 14.5, 38.0, 50),
                ("PAD", 11.2, 26.5, 50),
            ],
        )

    def test_production_model_download_replaces_missing_or_small_file(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "models" / "yolov8n.onnx"
            model_path.parent.mkdir()
            model_path.write_bytes(b"placeholder")

            def download(_url, filename):
                Path(filename).write_bytes(b"x" * app.YOLO_MODEL_MIN_BYTES)

            with (
                patch.object(app, "YOLO_MODEL_PATH", model_path),
                patch.object(app.urllib.request, "urlretrieve", side_effect=download) as retrieve,
            ):
                self.assertTrue(app.download_production_yolo_model())

            self.assertEqual(model_path.stat().st_size, app.YOLO_MODEL_MIN_BYTES)
            retrieve.assert_called_once_with(app.YOLO_MODEL_URL, unittest.mock.ANY)

    def test_production_model_download_failure_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "models" / "yolov8n.onnx"
            with (
                patch.object(app, "YOLO_MODEL_PATH", model_path),
                patch.object(
                    app.urllib.request,
                    "urlretrieve",
                    side_effect=OSError("network unavailable"),
                ),
                self.assertLogs(app.LOGGER, level="ERROR"),
            ):
                self.assertFalse(app.download_production_yolo_model())
            self.assertFalse(model_path.exists())

    async def test_control_profile_calibration_requires_one_shared_version(self):
        profiles = [
            {"control_type": "KBM", "base_avg_speed": 1.0, "max_legal_velocity": 2.0,
             "sample_count": 10, "speed_m2": 1.0},
            {"control_type": "PAD", "base_avg_speed": 1.0, "max_legal_velocity": 2.0,
             "sample_count": 10, "speed_m2": 1.0},
        ]
        baselines = [
            {"control_type": "KBM", "league_round": 1, "calibration_version": "k1-v1",
             "base_avg_speed": 1.0, "max_legal_velocity": 2.0, "sample_count": 10,
             "speed_m2": 1.0},
            {"control_type": "PAD", "league_round": 1, "calibration_version": "k1-v2",
             "base_avg_speed": 1.0, "max_legal_velocity": 2.0, "sample_count": 10,
             "speed_m2": 1.0},
        ]
        pool = SimpleNamespace(fetch=AsyncMock(side_effect=[profiles, baselines]))

        with self.assertRaisesRegex(RuntimeError, "versions must match"):
            await app._load_control_profiles(pool)


class ReportEmbedTests(unittest.TestCase):
    def test_embed_handles_null_and_zero_metrics_and_bounds_long_text(self):
        report = {
            "id": 999,
            "video_url": "https://example.test/video.mp4",
            "player_name": "Player",
            "player_id": 123,
            "control_type": "PAD",
            "avg_speed": 0,
            "magnitude_variance": None,
            "jitter_frequency": None,
            "is_anomalous": False,
            "raw_report_text": "clean report " * 1000,
        }

        embed = app._report_embed(report)

        self.assertIn("Średni ruch celownika (POV):** 0.00", embed.description)
        self.assertIn("Wariancja pędu celownika:** n/d", embed.description)
        self.assertIn("Częstotliwość mikro-szarpnięć:** n/d Hz", embed.description)
        self.assertLessEqual(len(embed.description), 4096)
        self.assertEqual(embed.description.count("```"), 2)
        self.assertIn("CZYSTO", embed.description)


class StreamDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_reads_in_1024_byte_chunks(self):
        response = FakeResponse([b"a" * 1024, b"b" * 512])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "clip.mp4"
            with patch.object(
                app.aiohttp, "ClientSession", return_value=FakeSession(response)
            ):
                await app.download_video("https://example.test/clip.mp4", target, 2048)

            self.assertEqual(target.stat().st_size, 1536)
            self.assertTrue(all(size == 1024 for size in response.content.read_sizes))
            self.assertEqual(response.content.read_sizes, [1024, 1024, 1024])

    async def test_download_stops_before_writing_chunk_that_exceeds_limit(self):
        response = FakeResponse([b"a" * 1024, b"b" * 1024, b"c" * 12])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "clip.mp4"
            with patch.object(
                app.aiohttp, "ClientSession", return_value=FakeSession(response)
            ):
                with self.assertRaises(app.FileTooLargeError):
                    await app.download_video(
                        "https://example.test/clip.mp4", target, max_size=1500
                    )

            self.assertEqual(target.stat().st_size, 1024)
            self.assertEqual(len(response.content.read_sizes), 2)


class AnalyzerTests(unittest.TestCase):
    def test_truncated_video_returns_no_alert_and_releases_decoder(self):
        capture = Mock()
        capture.read.return_value = (False, None)
        capture.isOpened.return_value = False
        with patch.object(app.cv2, "VideoCapture", return_value=capture):
            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(
                "truncated.mp4"
            )

        self.assertFalse(result["alert"])
        self.assertIn("Błąd", result["powod"])
        capture.release.assert_called_once()

    def test_truncated_mp4_header_is_rejected_by_real_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.mp4"
            path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00truncated")

            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(path)

        self.assertFalse(result["alert"])
        self.assertIn("powod", result)

    def test_black_video_and_zero_fps_do_not_raise_motion_alert(self):
        black_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [
            (True, black_frame),
            (True, black_frame),
            (True, black_frame),
            (False, None),
        ]
        capture.get.return_value = 0
        capture.isOpened.side_effect = [True, True, True, False]
        with patch.object(app.cv2, "VideoCapture", return_value=capture):
            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(
                "black-zero-fps.mp4"
            )

        self.assertFalse(result["alert"])
        self.assertEqual(result["raport"], "Brak krytycznych odchyleń.")
        capture.release.assert_called_once()

    def test_long_frame_gaps_are_skipped_instead_of_counted_as_motion(self):
        frames = [
            np.zeros((32, 32, 3), dtype=np.uint8),
            *[
                np.full((32, 32, 3), (index % 2) * 255, dtype=np.uint8)
                for index in range(7)
            ],
        ]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        capture.isOpened.return_value = True
        timestamps = iter([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000])

        def read_property(prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30
            return next(timestamps)

        capture.get.side_effect = read_property
        with (
            patch.object(app.cv2, "VideoCapture", return_value=capture),
            patch.object(app.cv2, "calcOpticalFlowFarneback") as optical_flow,
        ):
            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(
                "frame-drops.mp4"
            )

        self.assertFalse(result["alert"])
        optical_flow.assert_not_called()
        capture.release.assert_called_once()

    def test_opencv_decode_exception_is_reported_without_leaking_capture(self):
        capture = Mock()
        capture.read.side_effect = cv2.error("corrupt frame")
        with patch.object(app.cv2, "VideoCapture", return_value=capture):
            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(
                "broken.mp4"
            )

        self.assertFalse(result["alert"])
        capture.release.assert_called_once()

    def test_two_frame_spike_is_reported_as_lag_not_anomaly(self):
        frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(7)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        capture.isOpened.return_value = True
        timestamps = iter([0, 33, 66, 99, 132, 165, 198])

        def read_property(prop):
            return 30 if prop == cv2.CAP_PROP_FPS else next(timestamps)

        capture.get.side_effect = read_property
        flows = []
        for speed in (0, 0, 40, 40, 0, 0):
            flow = np.zeros((128, 128, 2), dtype=np.float32)
            flow[..., 0] = speed
            flows.append(flow)
        profile = {
            "PAD": {
                "base_avg_speed": 20.0,
                "max_legal_velocity": 50.0,
                "sample_count": 1,
                "speed_m2": 0.0,
            },
            "KBM": {
                "base_avg_speed": 20.0,
                "max_legal_velocity": 50.0,
                "sample_count": 1,
                "speed_m2": 0.0,
            },
        }
        with (
            patch.object(app.cv2, "VideoCapture", return_value=capture),
            patch.object(
                app.cv2, "calcOpticalFlowFarneback", side_effect=flows
            ),
        ):
            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(
                "lag.mp4", profile
            )

        self.assertFalse(result["alert"])
        self.assertTrue(result["lag_detected"])
        self.assertIn("Potencjalny Lag", result["raport"])

    def test_four_continuous_over_threshold_frames_raise_anomaly(self):
        frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(6)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        capture.isOpened.return_value = True
        timestamps = iter([0, 33, 66, 99, 132, 165])

        def read_property(prop):
            return 30 if prop == cv2.CAP_PROP_FPS else next(timestamps)

        capture.get.side_effect = read_property
        flows = []
        for _ in range(5):
            flow = np.zeros((128, 128, 2), dtype=np.float32)
            flow[..., 0] = 40
            flows.append(flow)
        profile = {
            name: {
                "base_avg_speed": 20.0,
                "max_legal_velocity": 50.0,
                "sample_count": 1,
                "speed_m2": 0.0,
            }
            for name in ("PAD", "KBM")
        }
        with (
            patch.object(app.cv2, "VideoCapture", return_value=capture),
            patch.object(
                app.cv2, "calcOpticalFlowFarneback", side_effect=flows
            ),
        ):
            result = app.SafeBallisticAnalyzer().procesuj_wideo_synchronicznie(
                "continuous.mp4", profile
            )

        self.assertTrue(result["alert"])
        self.assertIn("magnitude_variance", result)
        self.assertEqual(result["longest_anomaly_run"], 5)


class AsyncProcessingTests(unittest.IsolatedAsyncioTestCase):
    async def test_semaphore_reserves_capacity_before_discord_awaits(self):
        semaphore = asyncio.Semaphore(2)
        entered_downloads = 0
        both_entered = asyncio.Event()
        release_downloads = asyncio.Event()
        guild_id = 74
        archive_id = 703
        app.KANAL_ANTYCHEAT_MAGAZYN[guild_id] = archive_id

        async def wait_for_download_release(url, destination):
            nonlocal entered_downloads
            entered_downloads += 1
            if entered_downloads == 2:
                both_entered.set()
            await release_downloads.wait()
            Path(destination).write_bytes(b'video')

        messages = []
        for message_id in (1101, 1102, 1103):
            messages.append(
                SimpleNamespace(
                    author=SimpleNamespace(mention="@player", id=123),
                    channel=SimpleNamespace(id=archive_id, send=AsyncMock()),
                    content=f"https://example.test/{message_id}.mp4",
                    attachments=[],
                    id=message_id,
                    guild=SimpleNamespace(id=guild_id),
                    delete=AsyncMock(),
                )
            )

        tasks = []
        try:
            with (
                patch.object(app, "processing_semaphore", semaphore),
                patch.object(app, "database_pool", FakeDatabasePool()),
                patch.object(app, "download_video", side_effect=wait_for_download_release),
                patch.object(
                    app,
                    "analyze_video_async",
                    new_callable=AsyncMock,
                    return_value={
                        "alert": False,
                        "raport": "Brak krytycznych odchyleń.",
                        "avg_speed": 10.0,
                        "overall_avg_speed": 10.0,
                        "magnitude_variance": 0.0,
                        "jitter_frequency": 0.0,
                        "control_type": "KBM",
                    },
                ),
            ):
                tasks = [
                    asyncio.create_task(app.on_message(message))
                    for message in messages
                ]
                await asyncio.wait_for(both_entered.wait(), timeout=1)
                await tasks[2]

                self.assertEqual(entered_downloads, 2)
                self.assertIn("jest przetwarzane", messages[0].channel.send.await_args.args[0])
                self.assertIn("jest przetwarzane", messages[1].channel.send.await_args.args[0])
                messages[0].delete.assert_not_awaited()
                messages[1].delete.assert_not_awaited()
                messages[2].delete.assert_not_awaited()
                messages[2].channel.send.assert_awaited_once()
        finally:
            background_tasks = tuple(app._background_submission_tasks)
            release_downloads.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            app.KANAL_ANTYCHEAT_MAGAZYN.pop(guild_id, None)

    async def test_executor_analysis_does_not_block_event_loop(self):
        started = threading.Event()
        release_worker = threading.Event()

        class SlowAnalyzer:
            def procesuj_wideo_synchronicznie(self, path):
                started.set()
                if not release_worker.wait(timeout=3):
                    raise TimeoutError("test worker was not released")
                return {"alert": False, "raport": "done"}

        task = asyncio.create_task(app.analyze_video_async(SlowAnalyzer(), "30s.mp4"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            heartbeat = 0
            for _ in range(5):
                await asyncio.sleep(0.01)
                heartbeat += 1

            self.assertEqual(heartbeat, 5)
            self.assertFalse(task.done())
        finally:
            release_worker.set()

        self.assertEqual(await task, {"alert": False, "raport": "done"})

    async def test_temporary_file_is_removed_when_processing_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app.tempfile, "tempdir", directory):
                async def finish_download(url, destination):
                    Path(destination).write_bytes(b"partial video")

                message = SimpleNamespace(
                    author=SimpleNamespace(mention="@player", id=123),
                    channel=SimpleNamespace(id=700, send=AsyncMock()),
                    id=1001,
                    guild=SimpleNamespace(id=71),
                    delete=AsyncMock(),
                    attachments=[],
                )
                app.KANAL_ANTYCHEAT_MAGAZYN[71] = 700
                with (
                    patch.object(app, "database_pool", FakeDatabasePool()),
                    patch.object(app, "_video_url_from_message", return_value="https://example.test/video.mp4"),
                    patch.object(app, "download_video", side_effect=finish_download),
                    patch.object(
                        app,
                        "analyze_video_async",
                        side_effect=RuntimeError("simulated analysis failure"),
                    ),
                ):
                    await app.on_message(message)
                    background_tasks = tuple(app._background_submission_tasks)
                    if background_tasks:
                        await asyncio.gather(*background_tasks, return_exceptions=True)

            self.assertEqual(list(Path(directory).glob("verify_*.mp4")), [])
            message.delete.assert_not_awaited()
            self.assertIn("Nie udało się przeanalizować", message.channel.send.return_value.edit.await_args.kwargs["content"])
            app.KANAL_ANTYCHEAT_MAGAZYN.pop(71, None)

    async def test_processing_capacity_rejects_excess_work_without_temp_file(self):
        semaphore = asyncio.Semaphore(2)
        await semaphore.acquire()
        await semaphore.acquire()
        message = SimpleNamespace(
            author=SimpleNamespace(mention="@player", id=123),
            channel=SimpleNamespace(id=701, send=AsyncMock()),
            id=1002,
            guild=SimpleNamespace(id=72),
            delete=AsyncMock(),
            attachments=[],
        )
        app.KANAL_ANTYCHEAT_MAGAZYN[72] = 701
        try:
            with (
                patch.object(app, "processing_semaphore", semaphore),
                patch.object(app, "_video_url_from_message", return_value="https://example.test/clip.mp4"),
                patch.object(app.tempfile, "mkstemp") as mkstemp,
                patch.object(app, "download_video", new_callable=AsyncMock) as download,
            ):
                await app.on_message(message)

            mkstemp.assert_not_called()
            download.assert_not_awaited()
            message.delete.assert_not_awaited()
            message.channel.send.assert_awaited_once()
        finally:
            semaphore.release()
            semaphore.release()
            app.KANAL_ANTYCHEAT_MAGAZYN.pop(72, None)

    async def test_clean_and_anomalous_clips_both_dispatch_complete_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app.tempfile, "tempdir", directory):
                for anomalous in (False, True):
                    with self.subTest(anomalous=anomalous):
                        guild_id = 73
                        archive_id = 702
                        alert_id = 890
                        app.KANAL_ANTYCHEAT_MAGAZYN[guild_id] = archive_id
                        app.KANAL_ANTYCHEAT_ALERTY[guild_id] = alert_id

                        report_channel = SimpleNamespace(send=AsyncMock())
                        message = SimpleNamespace(
                            author=SimpleNamespace(mention="@player", id=123),
                            channel=SimpleNamespace(id=archive_id, send=AsyncMock()),
                            content="https://example.test/clips/clip%20one.mp4",
                            attachments=[],
                            id=1003,
                            guild=SimpleNamespace(id=guild_id),
                            delete=AsyncMock(),
                        )
                        result = {
                            "alert": anomalous,
                            "raport": "• Sekunda 1.00s (Wskaźnik pędu: 30.00)"
                            if anomalous
                            else "Brak krytycznych odchyleń.",
                            "avg_speed": 30.0 if anomalous else 10.0,
                            "overall_avg_speed": 15.0,
                            "magnitude_variance": 2.5,
                            "jitter_frequency": 1.0,
                            "control_type": "KBM",
                        }
                        database = FakeDatabasePool()
                        report_message = SimpleNamespace(id=4321)
                        report_channel.send.return_value = report_message
                        with (
                            patch.object(app, "database_pool", database),
                            patch.object(
                                app,
                                "download_video",
                                new_callable=AsyncMock,
                            ),
                            patch.object(
                                app,
                                "analyze_video_async",
                                new_callable=AsyncMock,
                                return_value=result,
                            ),
                            patch.object(
                                app.bot,
                                "get_channel",
                                return_value=report_channel,
                            ),
                            patch.object(app.bot, "add_view") as add_view,
                        ):
                            await app.on_message(message)
                            background_tasks = tuple(app._background_submission_tasks)
                            if background_tasks:
                                await asyncio.gather(
                                    *background_tasks, return_exceptions=True
                                )

                        message.delete.assert_awaited_once()
                        message.channel.send.assert_awaited_once()
                        self.assertIn(
                            "jest przetwarzane",
                            message.channel.send.await_args.args[0],
                        )
                        message.channel.send.return_value.edit.assert_awaited_once()
                        self.assertIn(
                            "został zapisany do weryfikacji",
                            message.channel.send.return_value.edit.await_args.kwargs["content"],
                        )
                        message.channel.send.return_value.delete.assert_awaited_once_with(
                            delay=15
                        )
                        report_channel.send.assert_awaited_once()
                        embed = report_channel.send.await_args.kwargs["embed"]
                        self.assertIn("clip one.mp4", embed.description)
                        self.assertIn("(ID: 123)", embed.description)
                        self.assertIn(
                            "[Otwórz wideo w przeglądarce](https://example.test/clips/clip%20one.mp4)",
                            embed.description,
                        )
                        self.assertIn(
                            result["raport"],
                            embed.description,
                        )
                        self.assertIsInstance(
                            report_channel.send.await_args.kwargs["view"],
                            app.PersistentAdminView,
                        )
                        self.assertEqual(
                            embed.title,
                            "⚠️ ALERT: WYKRYTO POTENCJALNĄ ANOMALIĘ RUCHU / FIZYKI"
                            if anomalous
                            else "✅ ANTY-CHEAT: MATERIAŁ W NORMIE",
                        )
                        self.assertIn(
                            "🔴 WYKRYTO KRYTYCZNE ODCHYLENIA OD NORMY"
                            if anomalous
                            else "🟢 CZYSTO (Ruch i fizyka w normie)",
                            embed.description,
                        )
                        self.assertEqual(
                            embed.color.value,
                            discord.Color.orange().value
                            if anomalous
                            else discord.Color.green().value,
                        )
                        self.assertEqual(database.report["message_id"], 4321)
                        add_view.assert_called_once()

        app.KANAL_ANTYCHEAT_MAGAZYN.pop(73, None)
        app.KANAL_ANTYCHEAT_ALERTY.pop(73, None)


class AdminViewTests(unittest.IsolatedAsyncioTestCase):
    def test_youtube_and_twitch_links_are_accepted_for_stream_resolution(self):
        for url in (
            "https://youtu.be/video-id",
            "https://www.youtube.com/watch?v=video-id",
            "https://www.twitch.tv/channel",
        ):
            message = SimpleNamespace(attachments=[], content=url)
            self.assertEqual(app._video_url_from_message(message), url)

    async def test_non_admin_gets_ephemeral_denial_without_deleting_archive(self):
        app.KANAL_ANTYCHEAT_ALERTY[44] = 888
        interaction = SimpleNamespace(
            guild_id=44,
            channel_id=888,
            guild=object(),
            user=SimpleNamespace(guild_permissions=SimpleNamespace(administrator=False)),
            response=SimpleNamespace(send_message=AsyncMock()),
            message=SimpleNamespace(embeds=[]),
        )
        view = app.PersistentAdminView()
        with patch.object(app.bot, "get_channel") as get_channel:
            await view.children[0].callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        self.assertTrue(interaction.response.send_message.await_args.kwargs["ephemeral"])
        get_channel.assert_not_called()
        app.KANAL_ANTYCHEAT_ALERTY.pop(44, None)

    async def test_view_has_four_restart_persistent_verdict_buttons(self):
        view = app.PersistentAdminView()

        self.assertEqual(
            [button.custom_id for button in view.children],
            ["ac_clean_kbm", "ac_clean_pad", "ac_cheat_detected", "ac_unknown_review"],
        )
        self.assertTrue(view.is_persistent())

    async def test_manual_review_does_not_change_control_profiles(self):
        database = FakeDatabasePool(report={
            "id": 84, "message_id": 5680, "report_status": "PENDING",
            "control_type": "PAD", "avg_speed": 42.0,
        })
        original_profiles = {
            name: profile.copy() for name, profile in database.profiles.items()
        }
        app.KANAL_ANTYCHEAT_ALERTY[60] = 892
        response = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
        interaction = SimpleNamespace(
            guild_id=60,
            channel_id=892,
            guild=object(),
            user=SimpleNamespace(guild_permissions=SimpleNamespace(administrator=True)),
            response=response,
            message=SimpleNamespace(
                id=5680, embeds=[discord.Embed(title="Pending review")]
            ),
        )

        with patch.object(app, "database_pool", database):
            await app.PersistentAdminView().children[3].callback(interaction)

        self.assertEqual(database.report["report_status"], "MANUAL_REVIEW")
        self.assertEqual(database.profiles, original_profiles)
        self.assertEqual(
            response.edit_message.await_args.kwargs["embed"].color.value,
            discord.Color.orange().value,
        )
        self.assertIn(
            "MANUALNA OBSERWACJA",
            response.edit_message.await_args.kwargs["embed"].title,
        )
        app.KANAL_ANTYCHEAT_ALERTY.pop(60, None)

    async def test_lock_timeout_notifies_staff_without_crashing(self):
        class LockError(Exception):
            sqlstate = "55P03"

        database = FakeDatabasePool(report={"id": 9, "message_id": 5678, "report_status": "PENDING"})
        database.fetchrow = AsyncMock(side_effect=LockError("lock not available"))
        app.KANAL_ANTYCHEAT_ALERTY[55] = 888
        response = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
        interaction = SimpleNamespace(
            guild_id=55,
            channel_id=888,
            guild=object(),
            user=SimpleNamespace(guild_permissions=SimpleNamespace(administrator=True)),
            response=response,
            message=SimpleNamespace(id=5678, embeds=[]),
        )
        with patch.object(app, "database_pool", database):
            await app.PersistentAdminView().children[0].callback(interaction)

        response.edit_message.assert_not_awaited()
        self.assertIn("zablokowany", response.send_message.await_args.args[0])

    async def test_clean_verdict_updates_profile_once_and_edits_alert(self):
        report = {
            "id": 81,
            "message_id": 5678,
            "report_status": "PENDING",
            "avg_speed": 20.0,
            "control_type": "PAD",
        }
        database = FakeDatabasePool(report=report)
        app.KANAL_ANTYCHEAT_ALERTY[55] = 888
        embed = discord.Embed(title="Pending review")
        response = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
        interaction = SimpleNamespace(
            guild_id=55,
            channel_id=888,
            guild=object(),
            user=SimpleNamespace(guild_permissions=SimpleNamespace(administrator=True)),
            response=response,
            message=SimpleNamespace(id=5678, embeds=[embed]),
        )

        with patch.object(app, "database_pool", database):
            await app.PersistentAdminView().children[0].callback(interaction)
            count_after_first_review = database.profiles["KBM"]["sample_count"]
            await app.PersistentAdminView().children[0].callback(interaction)

        self.assertEqual(database.report["report_status"], "VERIFIED_CLEAN")
        self.assertEqual(database.report["control_type"], "KBM")
        self.assertEqual(count_after_first_review, 51)
        self.assertEqual(database.profiles["KBM"]["sample_count"], 51)
        self.assertAlmostEqual(database.profiles["KBM"]["base_avg_speed"], 14.6078431373)
        response.edit_message.assert_awaited_once()
        self.assertIsNone(response.edit_message.await_args.kwargs["view"])
        self.assertEqual(response.send_message.await_count, 1)
        self.assertIn("CZYSTE", response.edit_message.await_args.kwargs["embed"].title)

    async def test_cheat_verdict_does_not_change_control_profiles(self):
        report = {
            "id": 82,
            "message_id": 5679,
            "report_status": "PENDING",
            "avg_speed": 99.0,
            "control_type": "PAD",
        }
        database = FakeDatabasePool(report=report)
        original_profiles = {
            name: profile.copy() for name, profile in database.profiles.items()
        }
        app.KANAL_ANTYCHEAT_ALERTY[58] = 891
        response = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
        interaction = SimpleNamespace(
            guild_id=58,
            channel_id=891,
            guild=object(),
            user=SimpleNamespace(guild_permissions=SimpleNamespace(administrator=True)),
            response=response,
            message=SimpleNamespace(
                id=5679, embeds=[discord.Embed(title="Pending review")]
            ),
        )

        with patch.object(app, "database_pool", database):
            await app.PersistentAdminView().children[2].callback(interaction)

        self.assertEqual(database.report["report_status"], "VERIFIED_CHEAT")
        self.assertEqual(database.profiles, original_profiles)
        self.assertEqual(response.edit_message.await_args.kwargs["embed"].color.value, discord.Color.red().value)
        app.KANAL_ANTYCHEAT_ALERTY.pop(58, None)

    async def test_pending_report_without_message_id_is_retried(self):
        report = {
            "id": 83,
            "message_id": None,
            "guild_id": 59,
            "player_id": 123,
            "player_name": "player",
            "video_url": "https://example.test/clip.mp4",
            "control_type": "PAD",
            "avg_speed": 10.0,
            "magnitude_variance": 0.0,
            "jitter_frequency": 0.0,
            "is_anomalous": False,
            "report_status": "PENDING",
            "raw_report_text": "Brak krytycznych odchyleń.",
        }
        database = FakeDatabasePool(report=report)
        database.pending_rows = [
            {"id": 83, "message_id": None, "guild_id": 59}
        ]
        channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=4322)))
        app.KANAL_ANTYCHEAT_ALERTY[59] = 890

        with (
            patch.object(app, "database_pool", database),
            patch.object(app.bot, "get_channel", return_value=channel),
            patch.object(app.bot, "add_view") as add_view,
        ):
            await app._recover_pending_report_alerts()

        channel.send.assert_awaited_once()
        self.assertEqual(database.report["message_id"], 4322)
        add_view.assert_called_once()
        app.KANAL_ANTYCHEAT_ALERTY.pop(59, None)


class DynamicChannelSetupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        app.KANAL_ANTYCHEAT_MAGAZYN.clear()
        app.KANAL_ANTYCHEAT_ALERTY.clear()

    async def test_creates_category_and_public_and_restricted_channels(self):
        bot_member = object()
        guild = FakeGuild(101, bot_member=bot_member)
        category = FakeCategory(200, app.ANTICHEAT_CATEGORY_NAME)
        guild.create_category.return_value = category

        async def create_category(name):
            self.assertEqual(name, app.ANTICHEAT_CATEGORY_NAME)
            guild.categories.append(category)
            return category

        guild.create_category.side_effect = create_category
        result = await app.setup_server_channels(guild)

        self.assertTrue(result)
        guild.create_category.assert_awaited_once_with(app.ANTICHEAT_CATEGORY_NAME)
        self.assertEqual(
            [channel.name for channel in guild.created_channels],
            [app.ANTICHEAT_ARCHIVE_CHANNEL, app.ANTICHEAT_ALERT_CHANNEL],
        )
        archive, alerts = guild.created_channels
        self.assertEqual(archive.set_permissions.await_count, 2)
        self.assertEqual(alerts.set_permissions.await_count, 2)
        self.assertIs(archive.overwrites[guild.default_role].view_channel, True)
        self.assertIs(archive.overwrites[guild.default_role].send_messages, True)
        self.assertIs(archive.overwrites[guild.default_role].read_message_history, False)
        self.assertIs(alerts.overwrites[guild.default_role].view_channel, False)
        self.assertIs(alerts.overwrites[guild.default_role].read_message_history, False)
        self.assertIs(alerts.overwrites[bot_member].view_channel, True)
        self.assertIs(alerts.overwrites[bot_member].send_messages, True)
        self.assertEqual(app.KANAL_ANTYCHEAT_MAGAZYN[101], archive.id)
        self.assertEqual(app.KANAL_ANTYCHEAT_ALERTY[101], alerts.id)

    async def test_reuses_existing_category_and_updates_channel_permissions(self):
        bot_member = object()
        everyone = object()
        category = FakeCategory(201, app.ANTICHEAT_CATEGORY_NAME)
        archive = FakeChannel(601, app.ANTICHEAT_ARCHIVE_CHANNEL)
        alerts = FakeChannel(602, app.ANTICHEAT_ALERT_CHANNEL)
        category.text_channels.extend([archive, alerts])
        guild = FakeGuild(102, categories=[category], bot_member=bot_member)
        guild.default_role = everyone

        result = await app.setup_server_channels(guild)

        self.assertTrue(result)
        guild.create_category.assert_not_awaited()
        guild.create_text_channel.assert_not_awaited()
        self.assertEqual(archive.set_permissions.await_count, 2)
        self.assertEqual(alerts.set_permissions.await_count, 2)
        self.assertIs(archive.overwrites[everyone].view_channel, True)
        self.assertIs(alerts.overwrites[everyone].view_channel, False)
        self.assertIs(alerts.overwrites[bot_member].send_messages, True)
        self.assertEqual(app.KANAL_ANTYCHEAT_MAGAZYN[102], archive.id)
        self.assertEqual(app.KANAL_ANTYCHEAT_ALERTY[102], alerts.id)

    async def test_existing_correct_channels_are_not_edited(self):
        guild = FakeGuild(103)
        bot_member = guild.me
        everyone = guild.default_role
        archive_overwrites = {
            everyone: discord.PermissionOverwrite(
                view_channel=True, send_messages=True, read_message_history=False
            ),
            bot_member: discord.PermissionOverwrite(
                view_channel=True,
                send_messages=True,
                read_message_history=True,
                manage_messages=True,
                attach_files=True,
            ),
        }
        alert_overwrites = {
            everyone: discord.PermissionOverwrite(
                view_channel=False, read_message_history=False
            ),
            bot_member: discord.PermissionOverwrite(
                view_channel=True,
                send_messages=True,
                read_message_history=True,
                manage_messages=True,
                attach_files=True,
            ),
        }
        archive = FakeChannel(603, app.ANTICHEAT_ARCHIVE_CHANNEL, archive_overwrites)
        alerts = FakeChannel(604, app.ANTICHEAT_ALERT_CHANNEL, alert_overwrites)
        category = FakeCategory(
            203,
            app.ANTICHEAT_CATEGORY_NAME,
            [archive, alerts],
        )
        guild.categories.append(category)

        result = await app.setup_server_channels(guild)

        self.assertTrue(result)
        archive.edit.assert_not_awaited()
        alerts.edit.assert_not_awaited()
        archive.set_permissions.assert_not_awaited()
        alerts.set_permissions.assert_not_awaited()
        self.assertEqual(app.KANAL_ANTYCHEAT_MAGAZYN[103], archive.id)
        self.assertEqual(app.KANAL_ANTYCHEAT_ALERTY[103], alerts.id)

    async def test_missing_bot_member_logs_and_returns_failure(self):
        guild = FakeGuild(104)
        guild.me = None
        archive = FakeChannel(610, app.ANTICHEAT_ARCHIVE_CHANNEL)
        alerts = FakeChannel(611, app.ANTICHEAT_ALERT_CHANNEL)
        category = FakeCategory(204, app.ANTICHEAT_CATEGORY_NAME, [archive, alerts])
        guild.categories.append(category)

        with self.assertLogs(app.LOGGER, level="ERROR"):
            result = await app.setup_server_channels(guild)

        self.assertFalse(result)
        guild.create_text_channel.assert_not_awaited()
        self.assertEqual(app.KANAL_ANTYCHEAT_MAGAZYN[104], archive.id)
        self.assertEqual(app.KANAL_ANTYCHEAT_ALERTY[104], alerts.id)

    async def test_forbidden_channel_creation_is_logged_without_raising(self):
        guild = FakeGuild(105)
        guild.create_category.side_effect = discord.Forbidden(
            SimpleNamespace(status=403, reason="Forbidden"),
            "Missing Manage Channels",
        )

        with self.assertLogs(app.LOGGER, level="ERROR"):
            result = await app.setup_server_channels(guild)

        self.assertFalse(result)

    async def test_http_error_during_channel_creation_is_logged_without_raising(self):
        guild = FakeGuild(106)
        category = FakeCategory(206, app.ANTICHEAT_CATEGORY_NAME)
        guild.categories.append(category)
        guild.create_text_channel.side_effect = discord.HTTPException(
            SimpleNamespace(status=500, reason="Server Error"),
            "Discord API error",
        )

        with self.assertLogs(app.LOGGER, level="ERROR"):
            result = await app.setup_server_channels(guild)

        self.assertFalse(result)

    async def test_unexpected_provisioning_error_is_logged_without_raising(self):
        guild = FakeGuild(107)
        guild.create_category.side_effect = RuntimeError("unexpected test failure")

        with self.assertLogs(app.LOGGER, level="ERROR"):
            result = await app.setup_server_channels(guild)

        self.assertFalse(result)

    async def test_on_ready_provisions_every_guild_and_registers_persistent_view_once(self):
        guilds = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        database = FakeDatabasePool()
        call_order = []

        async def run_in_thread(function):
            call_order.append("model")
            return function()

        async def initialize_database_with_order(_database):
            call_order.append("database")

        with (
            patch.object(app.bot, "_connection", SimpleNamespace(guilds=guilds)),
            patch.object(app, "setup_server_channels", new_callable=AsyncMock) as setup,
            patch.object(app.bot, "add_view") as add_view,
            patch.object(app, "database_pool", None),
            patch.object(app.asyncpg, "create_pool", new_callable=AsyncMock, return_value=database),
            patch.object(app, "download_production_yolo_model", return_value=True) as download_model,
            patch.object(app.asyncio, "to_thread", side_effect=run_in_thread),
            patch.object(
                app, "initialize_database", side_effect=initialize_database_with_order
            ),
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
        ):
            app._persistent_view_registered = False
            await app.on_ready()
            await app.on_ready()

        self.assertEqual(call_order, ["model", "database", "model", "database"])
        self.assertEqual(download_model.call_count, 2)
        self.assertEqual(setup.await_count, 4)
        self.assertEqual(
            [call.args[0] for call in setup.await_args_list],
            [guilds[0], guilds[1], guilds[0], guilds[1]],
        )
        add_view.assert_called_once()
        self.assertTrue(add_view.call_args.args[0].is_persistent())
        app._persistent_view_registered = False


if __name__ == "__main__":
    unittest.main()
