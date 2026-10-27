import asyncio
import functools
import io
import logging
import math
import os
import re
import tempfile
import urllib.request
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import aiohttp
import asyncpg
import cv2
import discord
import numpy as np
import yt_dlp
from discord.ext import commands
from dotenv import load_dotenv

import anticheat_report as report_tools
import anticheat_vision as vision

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("bpl.anticheat")

YOLO_MODEL_PATH = Path(vision.YOLO_MODEL_PATH)
YOLO_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
)
YOLO_MODEL_MIN_BYTES = 1_000_000


def download_production_yolo_model():
    """Ensure the configured YOLOv8 ONNX model is present and non-placeholder."""
    model_directory = YOLO_MODEL_PATH.parent
    temporary_path = None
    try:
        model_directory.mkdir(parents=True, exist_ok=True)
        if (
            YOLO_MODEL_PATH.is_file()
            and YOLO_MODEL_PATH.stat().st_size >= YOLO_MODEL_MIN_BYTES
        ):
            return True

        with tempfile.NamedTemporaryFile(
            dir=model_directory,
            prefix="yolov8n-",
            suffix=".download",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        urllib.request.urlretrieve(YOLO_MODEL_URL, str(temporary_path))
        if temporary_path.stat().st_size < YOLO_MODEL_MIN_BYTES:
            raise ValueError("Downloaded YOLOv8 ONNX model is smaller than 1 MB.")

        os.replace(temporary_path, YOLO_MODEL_PATH)
        temporary_path = None
        LOGGER.info("Verified production YOLOv8 model at %s", YOLO_MODEL_PATH)
        return True
    except Exception:
        LOGGER.exception("Could not verify or download the production YOLOv8 model")
        return False
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Could not remove temporary YOLO model download %s",
                    temporary_path,
                    exc_info=True,
                )


intents = discord.Intents.default()
intents.guilds = True
intents.message_content = True

KANAL_ANTYCHEAT_MAGAZYN = {}
KANAL_ANTYCHEAT_ALERTY = {}
ANTICHEAT_CATEGORY_NAME = "🏆 BPL ANTI-CHEAT"
ANTICHEAT_ARCHIVE_CHANNEL = "ac-magazyn-plikow"
ANTICHEAT_ALERT_CHANNEL = "ac-alerty-sedziowskie"
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
MAX_LIVE_ANALYSIS_SECONDS = float(os.getenv("MAX_LIVE_ANALYSIS_SECONDS", "3600"))
DOWNLOAD_CHUNK_BYTES = 1024
MIN_CONTINUOUS_ANOMALY_FRAMES = 4
CENTER_CROP_SIZE = 64

PROFILE_DEFAULTS = {
    "KBM": {"base_avg_speed": 14.5, "max_legal_velocity": 38.0, "sample_count": 50},
    "PAD": {"base_avg_speed": 11.2, "max_legal_velocity": 26.5, "sample_count": 50},
}

executor = ThreadPoolExecutor(max_workers=2)
processing_semaphore = asyncio.Semaphore(2)
database_pool = None
_persistent_view_registered = False
_registered_report_message_ids = set()
_report_locks = {}
_report_locks_guard = asyncio.Lock()
_background_submission_tasks = set()


class FileTooLargeError(Exception):
    """Raised when a streamed video exceeds its configured size limit."""


async def download_video(url, destination, max_size=MAX_FILE_SIZE_BYTES):
    timeout = aiohttp.ClientTimeout(total=60)
    total_bytes = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message="Video download failed",
                    headers=response.headers,
                )
            if response.content_length is not None and response.content_length > max_size:
                raise FileTooLargeError(f"Video exceeds the {max_size}-byte limit.")
            with open(destination, "wb") as output:
                while True:
                    chunk = await response.content.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_size:
                        raise FileTooLargeError(f"Video exceeds the {max_size}-byte limit.")
                    output.write(chunk)


def _initial_profile_stddev(profile):
    return (profile["max_legal_velocity"] - profile["base_avg_speed"]) / 3.0


# Each statement is idempotent, touches only anti-cheat-owned tables and runs
# on its own so no long multi-statement transaction ever holds locks.
SCHEMA_UPGRADE_STATEMENTS = (
    "ALTER TABLE reports DROP CONSTRAINT IF EXISTS reports_report_status_check",
    "ALTER TABLE reports ADD CONSTRAINT reports_report_status_check "
    "CHECK (report_status IN ('PENDING', 'VERIFIED_CLEAN', 'VERIFIED_CHEAT', 'MANUAL_REVIEW'))",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS offender_name TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS anomaly_type TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS anomaly_second DOUBLE PRECISION",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS anomaly_clock TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS detail_log TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS lag_detected BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS offender_avg_speed DOUBLE PRECISION",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS offender_jitter_hz DOUBLE PRECISION",
    "CREATE UNIQUE INDEX IF NOT EXISTS player_profiles_discord_id_unique_idx "
    "ON player_profiles (discord_id) WHERE discord_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS player_profiles_epic_key_unique_idx "
    "ON player_profiles (epic_key) WHERE epic_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS player_profiles_updated_idx "
    "ON player_profiles (updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS reports_pending_idx "
    "ON reports (id) WHERE report_status = 'PENDING'",
    "CREATE INDEX IF NOT EXISTS reports_player_created_idx "
    "ON reports (player_id, created_at DESC)",
)

# Welford update executed atomically inside the UPSERT (UPDATE expressions read
# the pre-update row), so concurrent verdicts cannot lose a sample.
_PROFILE_UPDATE_SET = """
    sample_count = player_profiles.sample_count + 1,
    speed_mean = player_profiles.speed_mean
        + ($2 - player_profiles.speed_mean) / (player_profiles.sample_count + 1),
    speed_m2 = player_profiles.speed_m2 + ($2 - player_profiles.speed_mean)
        * ($2 - (player_profiles.speed_mean
            + ($2 - player_profiles.speed_mean) / (player_profiles.sample_count + 1))),
    jitter_mean = player_profiles.jitter_mean
        + ($3 - player_profiles.jitter_mean) / (player_profiles.sample_count + 1),
    jitter_m2 = player_profiles.jitter_m2 + ($3 - player_profiles.jitter_mean)
        * ($3 - (player_profiles.jitter_mean
            + ($3 - player_profiles.jitter_mean) / (player_profiles.sample_count + 1))),
    variance_mean = player_profiles.variance_mean
        + ($4 - player_profiles.variance_mean) / (player_profiles.sample_count + 1),
    lag_clip_count = player_profiles.lag_clip_count + $5,
    updated_at = NOW()
"""
# Parameters: $1 key (discord id / epic key), $2 speed, $3 jitter, $4 variance,
# $5 lag flag (0/1), and for the epic-key variant $6 display nickname.
PROFILE_UPSERT_BY_DISCORD_ID = f"""
    INSERT INTO player_profiles
        (discord_id, sample_count, speed_mean, speed_m2, jitter_mean, jitter_m2,
         variance_mean, lag_clip_count)
    VALUES ($1, 1, $2, 0, $3, 0, $4, $5)
    ON CONFLICT (discord_id) WHERE discord_id IS NOT NULL DO UPDATE SET
    {_PROFILE_UPDATE_SET}
"""
PROFILE_UPSERT_BY_EPIC_KEY = f"""
    INSERT INTO player_profiles
        (epic_key, sample_count, speed_mean, speed_m2, jitter_mean, jitter_m2,
         variance_mean, lag_clip_count, epic_nickname)
    VALUES ($1, 1, $2, 0, $3, 0, $4, $5, $6)
    ON CONFLICT (epic_key) WHERE epic_key IS NOT NULL DO UPDATE SET
    {_PROFILE_UPDATE_SET}
"""
NICKNAME_REGISTER_SQL = """
    INSERT INTO player_profiles (epic_nickname, epic_key)
    VALUES ($1, $2)
    ON CONFLICT (epic_key) WHERE epic_key IS NOT NULL DO NOTHING
"""


async def initialize_database(pool):
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS gadgets_mechanics (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            speed_multiplier DOUBLE PRECISION NOT NULL CHECK (speed_multiplier > 0),
            duration_seconds DOUBLE PRECISION NOT NULL CHECK (duration_seconds > 0)
        );
        CREATE TABLE IF NOT EXISTS control_profiles (
            control_type TEXT PRIMARY KEY CHECK (control_type IN ('KBM', 'PAD')),
            base_avg_speed DOUBLE PRECISION NOT NULL,
            max_legal_velocity DOUBLE PRECISION NOT NULL,
            sample_count BIGINT NOT NULL CHECK (sample_count >= 0),
            speed_m2 DOUBLE PRECISION NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS control_profile_baselines (
            control_type TEXT NOT NULL CHECK (control_type IN ('KBM', 'PAD')),
            league_round INTEGER NOT NULL CHECK (league_round = 1),
            calibration_version TEXT NOT NULL,
            base_avg_speed DOUBLE PRECISION NOT NULL,
            max_legal_velocity DOUBLE PRECISION NOT NULL,
            sample_count BIGINT NOT NULL CHECK (sample_count >= 2),
            speed_m2 DOUBLE PRECISION NOT NULL CHECK (speed_m2 >= 0),
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (control_type, calibration_version)
        );
        ALTER TABLE control_profiles
            ADD COLUMN IF NOT EXISTS speed_m2 DOUBLE PRECISION NOT NULL DEFAULT 0;
        CREATE TABLE IF NOT EXISTS player_profiles (
            id BIGSERIAL PRIMARY KEY,
            discord_id BIGINT,
            epic_nickname TEXT,
            epic_key TEXT,
            sample_count BIGINT NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
            speed_mean DOUBLE PRECISION NOT NULL DEFAULT 0,
            speed_m2 DOUBLE PRECISION NOT NULL DEFAULT 0,
            jitter_mean DOUBLE PRECISION NOT NULL DEFAULT 0,
            jitter_m2 DOUBLE PRECISION NOT NULL DEFAULT 0,
            variance_mean DOUBLE PRECISION NOT NULL DEFAULT 0,
            lag_clip_count BIGINT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (discord_id IS NOT NULL OR epic_key IS NOT NULL)
        );
        CREATE TABLE IF NOT EXISTS reports (
            id BIGSERIAL PRIMARY KEY,
            message_id BIGINT UNIQUE,
            source_message_id BIGINT,
            guild_id BIGINT NOT NULL,
            player_id BIGINT NOT NULL,
            player_name TEXT NOT NULL,
            video_url TEXT NOT NULL,
            control_type TEXT NOT NULL CHECK (control_type IN ('KBM', 'PAD')),
            avg_speed DOUBLE PRECISION NOT NULL DEFAULT 0,
            magnitude_variance DOUBLE PRECISION NOT NULL DEFAULT 0,
            jitter_frequency DOUBLE PRECISION NOT NULL DEFAULT 0,
            is_anomalous BOOLEAN NOT NULL DEFAULT FALSE,
            report_status TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (report_status IN ('PENDING', 'VERIFIED_CLEAN', 'VERIFIED_CHEAT',
                                         'MANUAL_REVIEW')),
            raw_report_text TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    await pool.execute(
        "ALTER TABLE reports ADD COLUMN IF NOT EXISTS source_message_id BIGINT"
    )
    await pool.execute(
        """
        DROP INDEX IF EXISTS reports_source_message_id_idx;
        CREATE UNIQUE INDEX IF NOT EXISTS reports_source_message_id_unique_idx
        ON reports (source_message_id)
        """
    )
    for statement in SCHEMA_UPGRADE_STATEMENTS:
        await pool.execute(statement)
    await pool.executemany(
        """
        INSERT INTO gadgets_mechanics (name, speed_multiplier, duration_seconds)
        VALUES ($1, $2, $3)
        ON CONFLICT (name) DO NOTHING
        """,
        [
            ("Tactical_Sprint", 1.25, 3.0),
            ("Impulse_Grenade", 3.00, 1.8),
            ("Sliding", 1.40, 1.2),
        ],
    )
    for control_type, profile in PROFILE_DEFAULTS.items():
        stddev = _initial_profile_stddev(profile)
        await pool.execute(
            """
            INSERT INTO control_profiles
                (control_type, base_avg_speed, max_legal_velocity, sample_count, speed_m2)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (control_type) DO NOTHING
            """,
            control_type,
            profile["base_avg_speed"],
            profile["max_legal_velocity"],
            profile["sample_count"],
            profile["sample_count"] * stddev * stddev,
        )


async def _load_control_profiles(pool):
    records = await pool.fetch(
        """
        SELECT control_type, base_avg_speed, max_legal_velocity, sample_count, speed_m2
        FROM control_profiles
        """
    )
    profiles = {record["control_type"]: dict(record) for record in records}
    missing = set(PROFILE_DEFAULTS) - profiles.keys()
    if missing:
        raise RuntimeError(f"Database is missing control profiles: {', '.join(sorted(missing))}")
    baselines = await pool.fetch(
        """
        SELECT DISTINCT ON (control_type)
               control_type, league_round, calibration_version, base_avg_speed,
               max_legal_velocity, sample_count, speed_m2
        FROM control_profile_baselines
        WHERE league_round = 1
        ORDER BY control_type, recorded_at DESC
        """
    )
    baselines = [
        dict(baseline)
        for baseline in baselines
        if "calibration_version" in baseline
    ]
    if baselines and len(baselines) != len(PROFILE_DEFAULTS):
        raise RuntimeError(
            "Kolejka 1 calibration is incomplete; expected one baseline per control type."
        )
    if baselines:
        calibration_versions = {
            baseline["calibration_version"] for baseline in baselines
        }
        if len(calibration_versions) != 1:
            raise RuntimeError(
                "Kolejka 1 calibration versions must match across control types."
            )
        for baseline in baselines:
            profiles[baseline["control_type"]].update(
                calibration_baseline=dict(baseline),
                calibration_version=baseline["calibration_version"],
            )
    else:
        LOGGER.warning(
            "No Kolejka 1 anti-cheat calibration found; using legacy control profiles."
        )
    return profiles


def _center_crop(array):
    height, width = array.shape[:2]
    crop_height = min(CENTER_CROP_SIZE, height)
    crop_width = min(CENTER_CROP_SIZE, width)
    center_y, center_x = height // 2, width // 2
    top = max(0, center_y - crop_height // 2)
    left = max(0, center_x - crop_width // 2)
    return array[top : top + crop_height, left : left + crop_width]


def _profile_threshold(profile):
    count = max(int(profile["sample_count"]), 1)
    variance = max(float(profile["speed_m2"]) / count, 0.0)
    return float(profile["base_avg_speed"]) + 3.0 * math.sqrt(variance)


class SafeBallisticAnalyzer:
    def procesuj_wideo_synchronicznie(
        self,
        video_path,
        control_profiles=None,
        player_profile=None,
        roster=(),
        mechanics=None,
    ):
        cap = None
        multi = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            if not ret or first_frame is None or first_frame.size == 0:
                return {"alert": False, "powod": "Błąd odczytu struktury pliku wideo."}

            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 0 else 30.0
            multi = self._create_multi_analyzer(first_frame, fps, roster, mechanics)
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            first_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            prev_timestamp = (
                first_position_ms / 1000.0
                if np.isfinite(first_position_ms) and first_position_ms >= 0
                else 0.0
            )
            clip_start_timestamp = prev_timestamp
            frame_idx = 0
            center_speeds = []
            center_times = []
            overall_speeds = []
            center_vectors = []
            lag_detected = False
            live_limit_reached = False
            is_live_source = urlparse(str(video_path)).scheme in ("http", "https")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    LOGGER.warning("Ignoring an invalid decoded frame in %s", video_path)
                    break

                frame_idx += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (
                    position_ms / 1000.0
                    if np.isfinite(position_ms) and position_ms / 1000.0 > prev_timestamp
                    else prev_timestamp + (1.0 / fps)
                )
                elapsed = timestamp - prev_timestamp
                if elapsed > max(0.5, 3.0 / fps):
                    LOGGER.warning(
                        "Skipping optical-flow comparison across a %.2fs frame gap in %s",
                        elapsed,
                        video_path,
                    )
                    lag_detected = True
                    prev_gray = gray
                    prev_timestamp = timestamp
                    continue

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                overall_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                overall_speeds.append(float(np.mean(overall_magnitude)))
                center_flow = _center_crop(flow)
                center_magnitude, _ = cv2.cartToPolar(
                    center_flow[..., 0], center_flow[..., 1]
                )
                center_speeds.append(float(np.mean(center_magnitude)))
                center_times.append(timestamp)
                center_vectors.append(
                    (
                        float(np.mean(center_flow[..., 0])),
                        float(np.mean(center_flow[..., 1])),
                    )
                )
                multi = self._feed_multi(multi, frame, flow, timestamp, frame_idx)
                prev_gray = gray
                prev_timestamp = timestamp
                if (
                    is_live_source
                    and timestamp - clip_start_timestamp >= MAX_LIVE_ANALYSIS_SECONDS
                ):
                    live_limit_reached = True
                    break

            if not center_speeds:
                return {
                    "alert": False,
                    "raport": "Brak krytycznych odchyleń.",
                    "avg_speed": 0.0,
                    "overall_avg_speed": 0.0,
                    "magnitude_variance": 0.0,
                    "jitter_frequency": 0.0,
                    "control_type": "PAD",
                    "multi": None,
                }

            avg_speed = float(np.mean(center_speeds))
            magnitude_variance = float(np.var(center_speeds))
            duration = max(frame_idx / fps, 1.0 / fps)
            vector_array = np.asarray(center_vectors, dtype=np.float64)
            vector_steps = np.linalg.norm(np.diff(vector_array, axis=0), axis=1)
            direction_changes = 0
            if len(vector_array) > 1:
                dots = np.sum(vector_array[:-1] * vector_array[1:], axis=1)
                lengths = np.linalg.norm(vector_array[:-1], axis=1) * np.linalg.norm(
                    vector_array[1:], axis=1
                )
                direction_changes = int(np.count_nonzero((dots < 0) & (lengths > 0.25)))
            step_threshold = max(float(np.median(vector_steps)) * 2.5, 1.0) if len(vector_steps) else 1.0
            abrupt_changes = int(np.count_nonzero(vector_steps > step_threshold))
            jitter_frequency = float((direction_changes + abrupt_changes) / duration)
            rounded_vectors = np.round(vector_array * 2.0) / 2.0
            unique_ratio = len(np.unique(rounded_vectors, axis=0)) / max(len(rounded_vectors), 1)
            staircase_ratio = 1.0 - unique_ratio
            control_type = (
                "PAD"
                if jitter_frequency >= 5.0 or staircase_ratio >= 0.65
                else "KBM"
            )

            profiles = control_profiles or {
                name: {
                    **profile,
                    "speed_m2": profile["sample_count"]
                    * _initial_profile_stddev(profile) ** 2,
                }
                for name, profile in PROFILE_DEFAULTS.items()
            }
            profile = profiles[control_type]
            base_threshold = _profile_threshold(profile)
            threshold, threshold_source = report_tools.personalized_threshold(
                player_profile, base_threshold, profile.get("calibration_baseline")
            )
            min_run = report_tools.adaptive_min_run(
                player_profile, MIN_CONTINUOUS_ANOMALY_FRAMES
            )
            over_threshold = [speed > threshold for speed in center_speeds]
            longest_run = 0
            current_run = 0
            longest_run_end = -1
            longest_run_speeds = []
            current_run_speeds = []
            for index, (speed, above) in enumerate(zip(center_speeds, over_threshold)):
                if above:
                    current_run += 1
                    current_run_speeds.append(speed)
                else:
                    current_run = 0
                    current_run_speeds = []
                longest_run = max(longest_run, current_run)
                if current_run == longest_run:
                    longest_run_speeds = current_run_speeds.copy()
                    longest_run_end = index

            transient_spike = any(over_threshold) and longest_run < min_run
            sustained_avg_speed = (
                float(np.mean(longest_run_speeds)) if longest_run_speeds else 0.0
            )
            anomalous = longest_run >= min_run and sustained_avg_speed > threshold
            anomaly_second = None
            if anomalous:
                anomaly_second = float(center_times[longest_run_end - longest_run + 1])
                report = (
                    f"Utrzymująca się anomalia: {longest_run} klatek z rzędu; "
                    f"średni ruch celownika w serii {sustained_avg_speed:.2f}, "
                    f"średnia całego klipu {avg_speed:.2f}, próg {threshold:.2f} "
                    f"(profil {control_type})."
                )
            elif transient_spike or lag_detected:
                report = (
                    "Potencjalny Lag / Desynchronizacja sieciowa. "
                    "Krótkotrwały skok lub przerwa w czasie klatek nie został "
                    "zaklasyfikowany jako cheat."
                )
            elif avg_speed <= 0.01:
                report = "Brak krytycznych odchyleń."
            else:
                report = (
                    f"Brak krytycznych odchyleń. Średni ruch celownika "
                    f"{avg_speed:.2f}; próg {threshold:.2f} (profil {control_type})."
                )
            calibration_baseline = profile.get("calibration_baseline")
            baseline_z_score, baseline_probability = report_tools.calibration_statistics(
                calibration_baseline,
                sustained_avg_speed if anomalous else avg_speed,
            )
            if live_limit_reached:
                report += (
                    f" Analiza transmisji zakończona na skonfigurowanym limicie "
                    f"{MAX_LIVE_ANALYSIS_SECONDS:.0f} s."
                )

            multi_result = self._finalize_multi(
                multi, video_path, frame_idx, jitter_frequency
            )
            findings = list(multi_result.findings) if multi_result else []
            if findings:
                report += "\n" + "\n".join(
                    f"{finding.label} — {finding.offender or finding.entity_id} "
                    f"@ {finding.video_second:.1f}s"
                    + (f" (zegar {finding.clock})" if finding.clock else "")
                    for finding in findings[:5]
                )
            anomaly_clock = None
            if anomaly_second is not None and multi is not None:
                anomaly_clock = vision.format_clock(multi.timer.clock_at(anomaly_second))
            return {
                "alert": anomalous or bool(findings),
                "raport": report,
                "avg_speed": avg_speed,
                "overall_avg_speed": float(np.mean(overall_speeds)),
                "magnitude_variance": magnitude_variance,
                "jitter_frequency": jitter_frequency,
                "control_type": control_type,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "calibration_version": profile.get("calibration_version", "legacy-default"),
                "baseline_z_score": baseline_z_score,
                "baseline_probability": baseline_probability,
                "longest_anomaly_run": longest_run,
                "lag_detected": lag_detected or transient_spike,
                "anomaly_second": anomaly_second,
                "anomaly_clock": anomaly_clock,
                "live_limit_reached": live_limit_reached,
                "multi": multi_result,
            }
        except cv2.error:
            LOGGER.warning("OpenCV could not decode video %s", video_path, exc_info=True)
            return {"alert": False, "powod": "Błąd odczytu struktury pliku wideo."}
        finally:
            if cap is not None:
                cap.release()

    @staticmethod
    def _create_multi_analyzer(first_frame, fps, roster, mechanics):
        """Full-frame multi-player stage; failures must never block the legacy metrics."""
        try:
            height, width = first_frame.shape[:2]
            impulse = (mechanics or {}).get("Impulse_Grenade", (3.0, 1.8))
            return vision.MultiPlayerAnalyzer(
                fps,
                (width, height),
                hud=vision.HudReader(roster=roster),
                impulse_multiplier=float(impulse[0]),
                impulse_duration=float(impulse[1]),
            )
        except Exception:
            LOGGER.warning("Multi-player analysis disabled for this clip", exc_info=True)
            return None

    @staticmethod
    def _feed_multi(multi, frame, flow, timestamp, frame_idx):
        if multi is None:
            return None
        try:
            multi.process_frame(frame, flow, timestamp, frame_idx)
        except Exception:
            LOGGER.warning("Multi-player analysis aborted mid-clip", exc_info=True)
            return None
        return multi

    @staticmethod
    def _finalize_multi(multi, video_path, frame_count, jitter_hz):
        if multi is None:
            return None
        try:
            result = multi.finalize(jitter_hz)
            if urlparse(str(video_path)).scheme in ("http", "https"):
                events, audio_ok = [], False
            else:
                events, audio_ok = multi.audio_visual_events(video_path, frame_count)
            result.av_events = events
            result.audio_available = audio_ok
            return result
        except Exception:
            LOGGER.warning("Multi-player finalisation failed", exc_info=True)
            return None


async def analyze_video_async(analyzer, video_path, control_profiles=None, **context):
    loop = asyncio.get_running_loop()
    if control_profiles is None and not context:
        func = functools.partial(analyzer.procesuj_wideo_synchronicznie, str(video_path))
    else:
        kwargs_for_analyzer = {
            "control_profiles": control_profiles,
            "player_profile": context.get("player_profile"),
            "roster": context.get("roster", ()),
            "mechanics": context.get("mechanics"),
        }
        func = functools.partial(
            analyzer.procesuj_wideo_synchronicznie, str(video_path), **kwargs_for_analyzer
        )
    return await loop.run_in_executor(executor, func)


def _safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


async def _load_analysis_context(pool, player_id):
    """Optional per-player data; any failure degrades to the global thresholds."""
    context = {"player_profile": None, "roster": (), "mechanics": None}
    try:
        profile = await pool.fetchrow(
            """
            SELECT sample_count, speed_mean, speed_m2, lag_clip_count
            FROM player_profiles WHERE discord_id = $1
            """,
            player_id,
        )
        context["player_profile"] = dict(profile) if profile else None
        rows = await pool.fetch(
            """
            SELECT epic_nickname FROM player_profiles
            WHERE epic_nickname IS NOT NULL
            ORDER BY updated_at DESC LIMIT 2000
            """
        )
        context["roster"] = tuple(
            name for name in (_report_value(r, "epic_nickname", None) for r in rows) if name
        )
        rows = await pool.fetch(
            "SELECT name, speed_multiplier, duration_seconds FROM gadgets_mechanics"
        )
        context["mechanics"] = {
            r["name"]: (float(r["speed_multiplier"]), float(r["duration_seconds"]))
            for r in rows
            if _report_value(r, "name", None) and _report_value(r, "speed_multiplier", None)
        } or None
    except Exception:
        LOGGER.warning("Could not load player/roster context; using defaults", exc_info=True)
    return context


async def _register_discovered_nicknames(pool, names):
    """Auto-populate player_profiles with nicknames read from the HUD."""
    for name in list(names)[:12]:
        nickname = vision.clean_nickname(name)
        if nickname is None:
            continue
        try:
            await pool.execute(NICKNAME_REGISTER_SQL, nickname, nickname.lower())
        except Exception:
            LOGGER.warning("Could not register nickname %r", nickname, exc_info=True)


def _report_id_from_interaction(interaction):
    for embed in interaction.message.embeds:
        for field in embed.fields:
            if field.name == "ID raportu":
                try:
                    return int(field.value)
                except (TypeError, ValueError):
                    return None
    return None


async def setup_server_channels(guild):
    try:
        category = next(
            (item for item in guild.categories if item.name == ANTICHEAT_CATEGORY_NAME),
            None,
        )
        if category is None:
            category = await guild.create_category(ANTICHEAT_CATEGORY_NAME)
            LOGGER.info(
                "Created anti-cheat category '%s' in guild %s",
                ANTICHEAT_CATEGORY_NAME,
                guild.id,
            )

        existing_archive = next(
            (
                channel
                for channel in category.text_channels
                if channel.name == ANTICHEAT_ARCHIVE_CHANNEL
            ),
            None,
        )
        existing_alerts = next(
            (
                channel
                for channel in category.text_channels
                if channel.name == ANTICHEAT_ALERT_CHANNEL
            ),
            None,
        )
        if existing_archive is not None:
            KANAL_ANTYCHEAT_MAGAZYN[guild.id] = existing_archive.id
        if existing_alerts is not None:
            KANAL_ANTYCHEAT_ALERTY[guild.id] = existing_alerts.id

        bot_member = guild.me
        if bot_member is None:
            LOGGER.error(
                "Cannot provision anti-cheat channels in guild %s: bot member unavailable",
                guild.id,
            )
            return False

        everyone = guild.default_role
        archive_channel = next(
            (
                channel
                for channel in category.text_channels
                if channel.name == ANTICHEAT_ARCHIVE_CHANNEL
            ),
            None,
        )
        if archive_channel is None:
            archive_channel = await guild.create_text_channel(
                ANTICHEAT_ARCHIVE_CHANNEL, category=category
            )
            LOGGER.info("Created #%s in guild %s", ANTICHEAT_ARCHIVE_CHANNEL, guild.id)
        KANAL_ANTYCHEAT_MAGAZYN[guild.id] = archive_channel.id

        archive_permissions = (
            (
                everyone,
                discord.PermissionOverwrite(
                    view_channel=True, send_messages=True, read_message_history=False
                ),
            ),
            (
                bot_member,
                discord.PermissionOverwrite(
                    view_channel=True,
                    send_messages=True,
                    read_message_history=True,
                    manage_messages=True,
                    attach_files=True,
                ),
            ),
        )
        for target, overwrite in archive_permissions:
            if archive_channel.overwrites_for(target) != overwrite:
                await archive_channel.set_permissions(target, overwrite=overwrite)
                LOGGER.info(
                    "Updated permissions for #%s in guild %s",
                    ANTICHEAT_ARCHIVE_CHANNEL,
                    guild.id,
                )

        alert_channel = next(
            (
                channel
                for channel in category.text_channels
                if channel.name == ANTICHEAT_ALERT_CHANNEL
            ),
            None,
        )
        if alert_channel is None:
            alert_channel = await guild.create_text_channel(
                ANTICHEAT_ALERT_CHANNEL, category=category
            )
            LOGGER.info(
                "Created restricted #%s in guild %s", ANTICHEAT_ALERT_CHANNEL, guild.id
            )
        KANAL_ANTYCHEAT_ALERTY[guild.id] = alert_channel.id

        alert_permissions = (
            (
                everyone,
                discord.PermissionOverwrite(
                    view_channel=False, read_message_history=False
                ),
            ),
            (
                bot_member,
                discord.PermissionOverwrite(
                    view_channel=True,
                    send_messages=True,
                    read_message_history=True,
                    manage_messages=True,
                    attach_files=True,
                ),
            ),
        )
        for target, overwrite in alert_permissions:
            if alert_channel.overwrites_for(target) != overwrite:
                await alert_channel.set_permissions(target, overwrite=overwrite)
                LOGGER.info(
                    "Updated permissions for restricted #%s in guild %s",
                    ANTICHEAT_ALERT_CHANNEL,
                    guild.id,
                )

        KANAL_ANTYCHEAT_MAGAZYN[guild.id] = archive_channel.id
        KANAL_ANTYCHEAT_ALERTY[guild.id] = alert_channel.id
        LOGGER.info(
            "Anti-cheat channels ready in guild %s (archive=%s, alerts=%s)",
            guild.id,
            archive_channel.id,
            alert_channel.id,
        )
        return True
    except (discord.Forbidden, discord.HTTPException):
        LOGGER.exception("Discord rejected anti-cheat channel provisioning in guild %s", guild.id)
        return False
    except Exception:
        LOGGER.exception("Unexpected error provisioning anti-cheat channels in guild %s", guild.id)
        return False


LOCK_TIMEOUT_MS = 5000
LOCK_NOT_AVAILABLE_SQLSTATE = "55P03"


def _is_lock_timeout(error):
    return getattr(error, "sqlstate", None) == LOCK_NOT_AVAILABLE_SQLSTATE


async def _review_report(interaction, status, selected_control_type=None):
    if interaction.guild is None or not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ Tylko administrator ligi może zarządzać tym zgłoszeniem.",
            ephemeral=True,
        )
        return
    expected_channel = KANAL_ANTYCHEAT_ALERTY.get(interaction.guild_id)
    if expected_channel is None or interaction.channel_id != expected_channel:
        await interaction.response.send_message(
            "❌ Ten przycisk działa wyłącznie na kanale alertów anty-cheat.",
            ephemeral=True,
        )
        return
    if database_pool is None:
        await interaction.response.send_message(
            "❌ Baza raportów jest niedostępna. Spróbuj ponownie później.",
            ephemeral=True,
        )
        return

    try:
        async with database_pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT_MS}';")
                report = await connection.fetchrow(
                    "SELECT * FROM reports WHERE message_id = $1 FOR UPDATE",
                    interaction.message.id,
                )
                if report is None:
                    report_id = _report_id_from_interaction(interaction)
                    if report_id is not None:
                        report = await connection.fetchrow(
                            "SELECT * FROM reports WHERE id = $1 FOR UPDATE",
                            report_id,
                        )
                if report is None:
                    await interaction.response.send_message(
                        "❌ Nie znaleziono raportu w bazie danych.",
                        ephemeral=True,
                    )
                    return
                if report["report_status"] != "PENDING":
                    await interaction.response.send_message(
                        "ℹ️ Ten raport został już zweryfikowany.",
                        ephemeral=True,
                    )
                    return

                if status == "VERIFIED_CLEAN":
                    profile = await connection.fetchrow(
                        """
                        SELECT * FROM control_profiles
                        WHERE control_type = $1 FOR UPDATE
                        """,
                        selected_control_type,
                    )
                    old_count = int(profile["sample_count"])
                    old_mean = float(profile["base_avg_speed"])
                    new_count = old_count + 1
                    sample = float(report["avg_speed"])
                    delta = sample - old_mean
                    new_mean = ((old_mean * old_count) + sample) / new_count
                    new_m2 = float(profile["speed_m2"]) + delta * (sample - new_mean)
                    await connection.execute(
                        """
                        UPDATE control_profiles
                        SET base_avg_speed = $2, sample_count = $3, speed_m2 = $4
                        WHERE control_type = $1
                        """,
                        selected_control_type,
                        new_mean,
                        new_count,
                        max(new_m2, 0.0),
                    )
                    await connection.execute(
                        """
                        UPDATE reports
                        SET report_status = 'VERIFIED_CLEAN', control_type = $2
                        WHERE id = $1
                        """,
                        report["id"],
                        selected_control_type,
                    )
                    try:
                        async with connection.transaction():
                            await _store_clean_telemetry(connection, report)
                    except Exception:
                        LOGGER.exception(
                            "Could not store player telemetry for report %s; verdict kept",
                            report["id"],
                        )
                elif status == "VERIFIED_CHEAT":
                    await connection.execute(
                        "UPDATE reports SET report_status = 'VERIFIED_CHEAT' WHERE id = $1",
                        report["id"],
                    )
                else:
                    await connection.execute(
                        "UPDATE reports SET report_status = 'MANUAL_REVIEW' WHERE id = $1",
                        report["id"],
                    )
                updated = await connection.fetchrow(
                    "SELECT * FROM reports WHERE id = $1", report["id"]
                )
    except Exception as error:
        if _is_lock_timeout(error):
            LOGGER.warning(
                "Lock timeout while reviewing report on message %s", interaction.message.id
            )
            await interaction.response.send_message(
                "⏳ Raport jest chwilowo zablokowany przez inną transakcję "
                "(inny sędzia lub usługa). Spróbuj ponownie za chwilę.",
                ephemeral=True,
            )
            return
        LOGGER.exception("Failed to persist staff decision for report on message %s", interaction.message.id)
        await interaction.response.send_message(
            "❌ Nie udało się zapisać werdyktu. Raport pozostaje oczekujący.",
            ephemeral=True,
        )
        return

    embed = interaction.message.embeds[0] if interaction.message.embeds else discord.Embed()
    report_status = updated["report_status"]
    is_clean = report_status == "VERIFIED_CLEAN"
    is_manual_review = report_status == "MANUAL_REVIEW"
    embed.color = (
        discord.Color.green() if is_clean
        else discord.Color.orange() if is_manual_review
        else discord.Color.red()
    )
    embed.title = (
        f"✅ ZWERYFIKOWANO JAKO CZYSTE — {updated['control_type']}"
        if is_clean
        else "🟠 MANUALNA OBSERWACJA SĘDZIOWSKA"
        if is_manual_review
        else "🔴 POTWIERDZONY CHEAT"
    )
    embed.add_field(
        name="Werdykt sędziego",
        value=f"{updated['report_status']} · profil {updated['control_type']}",
        inline=False,
    )
    try:
        await interaction.response.edit_message(embed=embed, view=None)
    except discord.NotFound:
        LOGGER.warning("Alert message for report %s no longer exists", updated["id"])
        await _notify_ephemeral(
            interaction, "✅ Werdykt zapisany, ale wiadomość alertu została już usunięta."
        )
    except discord.HTTPException:
        LOGGER.exception("Could not update anti-cheat alert after staff review")
        await _notify_ephemeral(
            interaction, "✅ Werdykt zapisany, ale nie udało się zaktualizować wiadomości alertu."
        )


async def _notify_ephemeral(interaction, text):
    try:
        response = interaction.response
        is_done = getattr(response, "is_done", None)
        if callable(is_done) and is_done():
            await interaction.followup.send(text, ephemeral=True)
        else:
            await response.send_message(text, ephemeral=True)
    except (discord.HTTPException, AttributeError):
        LOGGER.warning("Could not deliver ephemeral staff notice", exc_info=True)


async def _store_clean_telemetry(connection, report):
    """Persist baselines only for judge-verified clean clips (Welford upsert)."""
    lag_flag = 1 if bool(_report_value(report, "lag_detected", False)) else 0
    player_id = _report_value(report, "player_id", None)
    if player_id is not None:
        await connection.execute(
            PROFILE_UPSERT_BY_DISCORD_ID,
            int(player_id),
            _safe_float(_report_value(report, "avg_speed", 0.0)),
            _safe_float(_report_value(report, "jitter_frequency", 0.0)),
            _safe_float(_report_value(report, "magnitude_variance", 0.0)),
            lag_flag,
        )
    offender = _report_value(report, "offender_name", None)
    offender_speed = _report_value(report, "offender_avg_speed", None)
    nickname = vision.clean_nickname(offender) if offender and offender_speed is not None else None
    if nickname is not None:
        await connection.execute(
            PROFILE_UPSERT_BY_EPIC_KEY,
            nickname.lower(),
            _safe_float(offender_speed),
            _safe_float(_report_value(report, "offender_jitter_hz", 0.0)),
            0.0,
            lag_flag,
            nickname,
        )


class PersistentAdminView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(
        label="Czysto - K&M",
        style=discord.ButtonStyle.success,
        custom_id="ac_clean_kbm",
        emoji="🟢",
    )
    async def clean_kbm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _review_report(interaction, "VERIFIED_CLEAN", "KBM")

    @discord.ui.button(
        label="Czysto - Pad",
        style=discord.ButtonStyle.primary,
        custom_id="ac_clean_pad",
        emoji="🔵",
    )
    async def clean_pad(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _review_report(interaction, "VERIFIED_CLEAN", "PAD")

    @discord.ui.button(
        label="Potwierdzony Cheat",
        style=discord.ButtonStyle.danger,
        custom_id="ac_cheat_detected",
        emoji="🔴",
    )
    async def confirmed_cheat(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _review_report(interaction, "VERIFIED_CHEAT")

    @discord.ui.button(
        label="Nie wiem / Obserwacja Sędziowska",
        style=discord.ButtonStyle.secondary,
        custom_id="ac_unknown_review",
        emoji="🤔",
    )
    async def manual_review(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _review_report(interaction, "MANUAL_REVIEW")


class AntiCheatBot(commands.Bot):
    async def close(self):
        global database_pool
        background_tasks = tuple(_background_submission_tasks)
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        if database_pool is not None:
            await database_pool.close()
            database_pool = None
        await super().close()


bot = AntiCheatBot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    global database_pool, _persistent_view_registered
    await asyncio.to_thread(download_production_yolo_model)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set to connect to Neon PostgreSQL.")
    if database_pool is None:
        database_pool = await asyncpg.create_pool(
            database_url, min_size=1, max_size=5, command_timeout=30, ssl="require"
        )
    await initialize_database(database_pool)
    for guild in bot.guilds:
        await setup_server_channels(guild)

    if not _persistent_view_registered:
        bot.add_view(PersistentAdminView())
        _persistent_view_registered = True
    await _recover_pending_report_alerts()
    LOGGER.info(
        "Anti-Cheat BPL is ready for manual review assistance in %s guild(s).",
        len(bot.guilds),
    )


def _video_url_from_message(message):
    if message.attachments:
        attachment = message.attachments[0]
        if (
            attachment.filename.lower().endswith((".mp4", ".mov"))
            and attachment.size <= MAX_FILE_SIZE_BYTES
        ):
            return attachment.url
        return None

    for url in re.findall(r"https?://[^\s<>]+", message.content):
        clean_url = url.rstrip(".,!?)]}")
        parsed = urlparse(clean_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.path.lower().endswith((".mp4", ".mov")) or (
            host == "youtu.be"
            or host == "youtube.com"
            or host.endswith(".youtube.com")
            or host == "twitch.tv"
            or host.endswith(".twitch.tv")
        ):
            return clean_url
    return None


def _resolve_video_source(video_url):
    """Return a direct media URL for supported video pages; do not download VODs."""
    host = (urlparse(video_url).hostname or "").lower()
    supported = (
        host == "youtu.be"
        or host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "twitch.tv"
        or host.endswith(".twitch.tv")
    )
    if not supported:
        return video_url, False
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "best[protocol^=https]/best",
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(options) as extractor:
        info = extractor.extract_info(video_url, download=False)
    direct_url = info.get("url") if info else None
    if not direct_url:
        formats = info.get("requested_formats", []) if info else []
        direct_url = formats[0].get("url") if formats else None
    if not direct_url:
        raise RuntimeError("yt-dlp did not return a playable stream URL.")
    return direct_url, True


def _filename_from_message(message, video_url):
    if message.attachments:
        return message.attachments[0].filename
    filename = unquote(urlparse(video_url).path.rsplit("/", 1)[-1])
    return filename or "video"


def _report_value(report, key, default=""):
    try:
        value = report[key]
    except (KeyError, TypeError):
        return default
    return default if value is None else value


def _finite_metric(report, key, precision):
    value = _report_value(report, key, None)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/d"
    if not math.isfinite(number):
        return "n/d"
    return f"{number:.{precision}f}"


def _single_line(value, limit):
    return " ".join(str(value).split())[:limit]


def _report_embed(report):
    anomalous = bool(_report_value(report, "is_anomalous", False))
    video_url = _single_line(_report_value(report, "video_url", ""), 900)
    try:
        filename = unquote(urlparse(video_url).path.rsplit("/", 1)[-1]) or "video"
    except ValueError:
        filename = "video"
    filename = _single_line(filename, 120)
    player_name = _single_line(_report_value(report, "player_name", "Nieznany"), 100)
    player_id = _single_line(_report_value(report, "player_id", "n/d"), 32)
    control_type = _single_line(_report_value(report, "control_type", "n/d"), 16)

    offender = _single_line(_report_value(report, "offender_name", "") or "—", 200)
    anomaly_type = _single_line(
        _report_value(report, "anomaly_type", "") or "Brak wykrytej anomalii", 200
    )
    clock = _single_line(_report_value(report, "anomaly_clock", ""), 40)
    second_text = _finite_metric(report, "anomaly_second", 2)

    raw_report = str(_report_value(report, "raw_report_text", "Brak szczegółów raportu."))
    raw_report = raw_report.replace("```", "'''")[:1800]

    numeric_stats = (
        f"**Średni ruch celownika (POV):** {_finite_metric(report, 'avg_speed', 2)}\n"
        f"**Wariancja pędu celownika:** {_finite_metric(report, 'magnitude_variance', 4)}\n"
        f"**Częstotliwość mikro-szarpnięć:** {_finite_metric(report, 'jitter_frequency', 2)} Hz\n"
    )
    if offender != "—" and offender != "POV (kamera zgłaszającego)":
        offender_speed = _finite_metric(report, "offender_avg_speed", 2)
        offender_jitter = _finite_metric(report, "offender_jitter_hz", 2)
        numeric_stats += f"**Średnia prędkość oskarżonego:** {offender_speed}\n"
        numeric_stats += f"**Jitter oskarżonego:** {offender_jitter} Hz\n"

    status_text = (
        "🔴 WYKRYTO KRYTYCZNE ODCHYLENIA OD NORMY"
        if anomalous
        else "🟢 CZYSTO (Ruch i fizyka w normie)"
    )
    video_link = quote(video_url, safe=":/?&=%#[]!$'*+,;@")

    embed = discord.Embed(
        title=(
            "⚠️ ALERT: WYKRYTO POTENCJALNĄ ANOMALIĘ RUCHU / FIZYKI"
            if anomalous
            else "✅ ANTY-CHEAT: MATERIAŁ W NORMIE"
        ),
        color=discord.Color.orange() if anomalous else discord.Color.green(),
    )

    footer_note = (
        "*Powyższe dane stanowią wynik automatycznej analizy i asysty dla sędziów BPL. "
        "Nie stanowią ostatecznego wyroku.*"
    )
    embed.description = (
        "**Materiały do analizy i weryfikacji sędziowskiej:**\n\n"
        f"**Plik wideo:** `{filename}`\n"
        f"**Zgłaszający (POV):** {player_name} (ID: {player_id})\n"
        f"**Wykryty Przeciwnik:** ` {offender} `\n"
        f"**Typ Wykrytej Anomalii:** `{anomaly_type}`\n"
        f"**Czas zdarzenia na wideo:** "
        + (f"`{second_text}s`" if second_text != "n/d" else "`n/d`")
        + (f" · ⏱️ Zegar rundy: `{clock}`" if clock else "")
        + "\n"
        f"**Profil sterowania (POV):** `{control_type}`\n\n"
        f"**Metryki szczegółowe:**\n{numeric_stats}\n"
        f"**Status bota:** {status_text}\n\n"
        f"**Dowód wideo:** [Otwórz wideo w przeglądarce]({video_link})\n\n"
        "**Logi z analizatora wizyjnego:**\n"
        f"```text\n{raw_report}\n```\n"
        f"{footer_note}"
    )

    if len(embed.description) > 4096:
        overflow = len(embed.description) - 4096
        raw_report = raw_report[: max(0, len(raw_report) - overflow - 4)]
        embed.description = (
            embed.description.split("```text\n", 1)[0]
            + "```text\n"
            + raw_report
            + "\n```\n"
            + footer_note
        )

    embed.add_field(
        name="ID raportu w bazie Neon",
        value=f"` #{_report_value(report, 'id', 'n/d')} `"[:1024],
        inline=True,
    )
    embed.set_footer(text=f"BPL Anti-Cheat Assistant · Raport #{_report_value(report, 'id', 'n/d')}")
    return embed


@asynccontextmanager
async def _lock_source_message(message_id):
    async with _report_locks_guard:
        entry = _report_locks.get(message_id)
        if entry is None:
            lock, references = asyncio.Lock(), 0
        else:
            lock, references = entry
        _report_locks[message_id] = (lock, references + 1)
    try:
        async with lock:
            yield
    finally:
        async with _report_locks_guard:
            lock, references = _report_locks[message_id]
            if references == 1:
                del _report_locks[message_id]
            else:
                _report_locks[message_id] = (lock, references - 1)


def _detail_attachment(report):
    report_id = _report_value(report, "id", "n/d")
    text = str(_report_value(report, "detail_log", "") or "")
    if not text:
        text = (
            f"BPL ANTI-CHEAT ANALYSIS - REPORT {report_id}\n\n"
            f"{_report_value(report, 'raw_report_text', 'No detailed log stored.')}\n"
        )
    text = text.replace(report_tools.REPORT_ID_TOKEN, str(report_id))
    return discord.File(
        io.BytesIO(text.encode("utf-8")), filename=f"Detailed_Analysis_{report_id}.txt"
    )


async def _dispatch_report(report_id, alert_channel):
    async with database_pool.acquire() as connection:
        async with connection.transaction():
            report = await connection.fetchrow(
                "SELECT * FROM reports WHERE id = $1 FOR UPDATE", report_id
            )
            if report is None:
                raise RuntimeError(f"Report {report_id} disappeared before dispatch.")
            if report["message_id"] is not None:
                return report["message_id"]
            alert_message = await alert_channel.send(
                embed=_report_embed(report),
                file=_detail_attachment(report),
                view=PersistentAdminView(),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await connection.execute(
                "UPDATE reports SET message_id = $2 WHERE id = $1",
                report_id,
                alert_message.id,
            )
    bot.add_view(PersistentAdminView(), message_id=alert_message.id)
    _registered_report_message_ids.add(alert_message.id)
    return alert_message.id


async def _recover_pending_report_alerts():
    rows = await database_pool.fetch(
        """
        SELECT id, message_id, guild_id
        FROM reports
        WHERE report_status = 'PENDING'
        """
    )
    for row in rows:
        message_id = row["message_id"]
        if message_id is not None:
            if message_id not in _registered_report_message_ids:
                bot.add_view(PersistentAdminView(), message_id=message_id)
                _registered_report_message_ids.add(message_id)
            continue

        alert_channel_id = KANAL_ANTYCHEAT_ALERTY.get(row["guild_id"])
        alert_channel = bot.get_channel(alert_channel_id) if alert_channel_id else None
        if alert_channel is None:
            LOGGER.error(
                "Cannot recover pending report %s: alert channel %s unavailable",
                row["id"],
                alert_channel_id,
            )
            continue
        try:
            await _dispatch_report(row["id"], alert_channel)
        except Exception:
            LOGGER.exception("Could not recover alert for pending report %s", row["id"])


async def _send_submission_notice(message, content, delete_after=15):
    try:
        await message.channel.send(content, delete_after=delete_after)
    except discord.HTTPException:
        LOGGER.exception(
            "Could not send submission status for source message %s",
            message.id,
        )
        return False
    return True


async def _send_submission_status(message, content):
    try:
        return await message.channel.send(content)
    except discord.HTTPException:
        LOGGER.exception(
            "Could not send submission status for source message %s",
            message.id,
        )
        return None


async def _update_submission_status(
    status_message, source_message_id, content, delete_after=None
):
    try:
        await status_message.edit(content=content)
        if delete_after is not None:
            await status_message.delete(delay=delete_after)
    except discord.HTTPException:
        LOGGER.exception(
            "Could not update submission status for source message %s",
            source_message_id,
        )
        return False
    return True


@bot.event
async def on_message(message):
    archive_channel_id = (
        KANAL_ANTYCHEAT_MAGAZYN.get(message.guild.id) if message.guild else None
    )
    if message.author == bot.user or message.channel.id != archive_channel_id:
        return

    video_url = _video_url_from_message(message)
    if video_url is None:
        return
    if processing_semaphore.locked():
        LOGGER.warning("Rejecting video message %s: processing capacity is full", message.id)
        await _send_submission_notice(
            message,
            "W tej chwili przetwarzane są inne zgłoszenia. "
            "Wiadomość z wideo pozostała na kanale — spróbuj ponownie za chwilę."
        )
        return
    if database_pool is None:
        LOGGER.error(
            "Cannot accept video message %s: database connection is not initialized",
            message.id,
        )
        await _send_submission_notice(
            message,
            "Nie można teraz przyjąć zgłoszenia z powodu niedostępnej bazy danych. "
            "Wiadomość z wideo pozostała na kanale — spróbuj ponownie później."
        )
        return

    await processing_semaphore.acquire()
    task_created = False
    try:
        status_message = await _send_submission_status(
            message,
            f"💚 Dziękujemy, {message.author.mention}! "
            "Twoje zgłoszenie jest przetwarzane. Prosimy pozostawić tę wiadomość na kanale."
        )
        if status_message is None:
            return

        filename = _filename_from_message(message, video_url)
        task = asyncio.create_task(
            _process_submission(message, video_url, filename, status_message),
            name=f"anticheat-report-{message.id}",
        )
        _background_submission_tasks.add(task)

        def discard_completed_task(completed_task):
            _background_submission_tasks.discard(completed_task)
            if completed_task.cancelled():
                return
            error = completed_task.exception()
            if error is not None:
                LOGGER.error(
                    "Background submission task failed for source message %s",
                    message.id,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(discard_completed_task)
        task_created = True
    except Exception:
        LOGGER.exception(
            "Could not schedule background processing for source message %s",
            message.id,
        )
        await _send_submission_notice(
            message,
            "Nie udało się rozpocząć przetwarzania zgłoszenia. "
            "Wiadomość z wideo pozostała na kanale — spróbuj ponownie później."
        )
    finally:
        if not task_created:
            processing_semaphore.release()


async def _process_submission(message, video_url, filename, status_message):
    temp_path = None
    report_persisted = False
    needs_dispatch = False
    try:
        async with _lock_source_message(message.id):
            async with database_pool.acquire() as connection:
                existing = await connection.fetchrow(
                    """
                    SELECT id, message_id, report_status
                    FROM reports
                    WHERE source_message_id = $1
                    """,
                    message.id,
                )
            if existing is not None:
                report_id = existing["id"]
                report_persisted = True
                needs_dispatch = (
                    existing["report_status"] == "PENDING"
                    and existing["message_id"] is None
                )
            else:
                profiles = await _load_control_profiles(database_pool)
                analysis_context = await _load_analysis_context(
                    database_pool, message.author.id
                )
                analysis_source, is_stream = await asyncio.to_thread(
                    _resolve_video_source, video_url
                )
                if not is_stream:
                    fd, temp_path = tempfile.mkstemp(prefix="verify_", suffix=".mp4")
                    os.close(fd)
                    await download_video(analysis_source, temp_path)
                    analysis_source = temp_path
                result = await analyze_video_async(
                    SafeBallisticAnalyzer(), analysis_source, profiles, **analysis_context
                )
                if result.get("powod"):
                    raise RuntimeError(result["powod"])

                offender, anomaly_type, anomaly_second, anomaly_clock = (
                    report_tools.summarize_anomaly(result)
                )
                detail_log = report_tools.build_detailed_analysis(
                    result,
                    {
                        "filename": filename,
                        "player_name": str(message.author),
                        "player_id": message.author.id,
                    },
                )
                headline = report_tools.headline_finding(result)
                offender_stats = None
                if headline is not None and headline.offender:
                    offender_stats = next(
                        (
                            entity
                            for entity in result["multi"].entities
                            if entity["entity_id"] == headline.entity_id
                        ),
                        None,
                    )
                multi = result.get("multi")
                discovered = list(getattr(multi, "discovered_nicknames", []) or [])
                if headline is not None and headline.offender:
                    discovered.append(headline.offender)
                await _register_discovered_nicknames(database_pool, discovered)

                async with database_pool.acquire() as connection:
                    report_id = await connection.fetchval(
                        """
                        INSERT INTO reports (
                            source_message_id, guild_id, player_id, player_name, video_url,
                            control_type, avg_speed, magnitude_variance, jitter_frequency,
                            is_anomalous, report_status, raw_report_text,
                            offender_name, anomaly_type, anomaly_second, anomaly_clock,
                            detail_log, lag_detected, offender_avg_speed, offender_jitter_hz
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'PENDING', $11,
                                $12, $13, $14, $15, $16, $17, $18, $19)
                        ON CONFLICT (source_message_id) DO NOTHING
                        RETURNING id
                        """,
                        message.id,
                        message.guild.id,
                        message.author.id,
                        str(message.author),
                        video_url,
                        result["control_type"],
                        result["avg_speed"],
                        result["magnitude_variance"],
                        result["jitter_frequency"],
                        result["alert"],
                        (
                            f"{result['raport']}\n"
                            f"Średnia prędkość globalna kadru: "
                            f"{result.get('overall_avg_speed', 0.0):.2f}."
                        ),
                        offender,
                        anomaly_type,
                        anomaly_second,
                        anomaly_clock,
                        detail_log,
                        bool(result.get("lag_detected", False)),
                        offender_stats["avg_speed"] if offender_stats else None,
                        offender_stats["jitter_hz"] if offender_stats else None,
                    )
                    if report_id is None:
                        report_id = await connection.fetchval(
                            "SELECT id FROM reports WHERE source_message_id = $1",
                            message.id,
                        )
                if report_id is None:
                    raise RuntimeError(
                        f"Could not persist or retrieve report for source message {message.id}."
                    )
                report_persisted = True
                needs_dispatch = True

            if needs_dispatch:
                alert_channel_id = KANAL_ANTYCHEAT_ALERTY.get(message.guild.id)
                alerts_channel = (
                    bot.get_channel(alert_channel_id) if alert_channel_id else None
                )
                if alerts_channel is None:
                    LOGGER.error(
                        "Report %s is persisted but alert channel %s is unavailable; "
                        "it will be retried when the bot reconnects",
                        report_id,
                        alert_channel_id,
                    )
                    await _update_submission_status(
                        status_message,
                        message.id,
                        "Zgłoszenie zostało zapisane, ale nie udało się powiadomić sędziów. "
                        "System ponowi wysyłkę; wiadomość z wideo pozostała na kanale.",
                    )
                    return
                await _dispatch_report(report_id, alerts_channel)

            status_updated = await _update_submission_status(
                status_message,
                message.id,
                f"💚 Dziękujemy za zgłoszenie, {message.author.mention}! "
                f"Plik {filename} został zapisany do weryfikacji.",
                delete_after=15,
            )
            if not status_updated:
                return
            try:
                await message.delete()
            except discord.HTTPException:
                LOGGER.exception(
                    "Could not remove source message %s after report %s was persisted",
                    message.id,
                    report_id,
                )
    except FileTooLargeError:
        LOGGER.warning("Report processing failed: video too large (message %s)", message.id)
        await _update_submission_status(
            status_message,
            message.id,
            "Nie udało się przyjąć zgłoszenia: plik przekracza dozwolony limit. "
            "Wiadomość nie została usunięta."
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        LOGGER.exception("Report processing failed for message %s", message.id)
        await _update_submission_status(
            status_message,
            message.id,
            "Nie udało się przetworzyć zgłoszenia z powodu błędu pobierania lub pliku. "
            "Wiadomość pozostała na kanale — spróbuj ponownie później."
        )
    except Exception:
        LOGGER.exception("Failed to process or dispatch report for message %s", message.id)
        if report_persisted:
            status_text = (
                "Zgłoszenie zostało zapisane, ale wystąpił problem z wysłaniem "
                "powiadomienia do sędziów. System ponowi wysyłkę; nie wysyłaj "
                "ponownie tego samego zgłoszenia. Wiadomość z wideo pozostała na kanale."
            )
        else:
            status_text = (
                "Nie udało się przeanalizować lub zapisać zgłoszenia. "
                "Wiadomość pozostała na kanale — spróbuj ponownie później."
            )
        await _update_submission_status(
            status_message=status_message,
            source_message_id=message.id,
            content=status_text,
        )
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not remove temporary video %s", temp_path)
        processing_semaphore.release()


if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN must be set.")
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL must be set to connect to Neon PostgreSQL.")
    bot.run(token)
