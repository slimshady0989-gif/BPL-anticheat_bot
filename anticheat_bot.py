import asyncio
import logging
import math
import os
import re
import tempfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import aiohttp
import asyncpg
import cv2
import discord
import numpy as np
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("bpl.anticheat")

intents = discord.Intents.default()
intents.guilds = True
intents.message_content = True

KANAL_ANTYCHEAT_MAGAZYN = {}
KANAL_ANTYCHEAT_ALERTY = {}
ANTICHEAT_CATEGORY_NAME = "🏆 BPL ANTI-CHEAT"
ANTICHEAT_ARCHIVE_CHANNEL = "ac-magazyn-plikow"
ANTICHEAT_ALERT_CHANNEL = "ac-alerty-sedziowskie"
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
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
        ALTER TABLE control_profiles
            ADD COLUMN IF NOT EXISTS speed_m2 DOUBLE PRECISION NOT NULL DEFAULT 0;
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
                CHECK (report_status IN ('PENDING', 'VERIFIED_CLEAN', 'VERIFIED_CHEAT')),
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
    def procesuj_wideo_synchronicznie(self, video_path, control_profiles=None):
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            if not ret or first_frame is None or first_frame.size == 0:
                return {"alert": False, "powod": "Błąd odczytu struktury pliku wideo."}

            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 0 else 30.0
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            first_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            prev_timestamp = (
                first_position_ms / 1000.0
                if np.isfinite(first_position_ms) and first_position_ms >= 0
                else 0.0
            )
            frame_idx = 0
            center_speeds = []
            overall_speeds = []
            center_vectors = []
            lag_detected = False

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
                center_vectors.append(
                    (
                        float(np.mean(center_flow[..., 0])),
                        float(np.mean(center_flow[..., 1])),
                    )
                )
                prev_gray = gray
                prev_timestamp = timestamp

            if not center_speeds:
                return {
                    "alert": False,
                    "raport": "Brak krytycznych odchyleń.",
                    "avg_speed": 0.0,
                    "overall_avg_speed": 0.0,
                    "magnitude_variance": 0.0,
                    "jitter_frequency": 0.0,
                    "control_type": "PAD",
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
            threshold = _profile_threshold(profile)
            over_threshold = [speed > threshold for speed in center_speeds]
            longest_run = 0
            current_run = 0
            longest_run_speeds = []
            current_run_speeds = []
            for speed, above in zip(center_speeds, over_threshold):
                if above:
                    current_run += 1
                    current_run_speeds.append(speed)
                else:
                    current_run = 0
                    current_run_speeds = []
                longest_run = max(longest_run, current_run)
                if current_run == longest_run:
                    longest_run_speeds = current_run_speeds.copy()

            transient_spike = any(over_threshold) and longest_run < MIN_CONTINUOUS_ANOMALY_FRAMES
            sustained_avg_speed = (
                float(np.mean(longest_run_speeds)) if longest_run_speeds else 0.0
            )
            anomalous = (
                longest_run >= MIN_CONTINUOUS_ANOMALY_FRAMES
                and sustained_avg_speed > threshold
            )
            if anomalous:
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
            return {
                "alert": anomalous,
                "raport": report,
                "avg_speed": avg_speed,
                "overall_avg_speed": float(np.mean(overall_speeds)),
                "magnitude_variance": magnitude_variance,
                "jitter_frequency": jitter_frequency,
                "control_type": control_type,
                "threshold": threshold,
                "longest_anomaly_run": longest_run,
                "lag_detected": lag_detected or transient_spike,
            }
        except cv2.error:
            LOGGER.warning("OpenCV could not decode video %s", video_path, exc_info=True)
            return {"alert": False, "powod": "Błąd odczytu struktury pliku wideo."}
        finally:
            if cap is not None:
                cap.release()


async def analyze_video_async(analyzer, video_path, control_profiles=None):
    loop = asyncio.get_running_loop()
    if control_profiles is None:
        return await loop.run_in_executor(
            executor, analyzer.procesuj_wideo_synchronicznie, str(video_path)
        )
    return await loop.run_in_executor(
        executor,
        analyzer.procesuj_wideo_synchronicznie,
        str(video_path),
        control_profiles,
    )


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
                else:
                    await connection.execute(
                        "UPDATE reports SET report_status = 'VERIFIED_CHEAT' WHERE id = $1",
                        report["id"],
                    )
                updated = await connection.fetchrow(
                    "SELECT * FROM reports WHERE id = $1", report["id"]
                )
    except Exception:
        LOGGER.exception("Failed to persist staff decision for report on message %s", interaction.message.id)
        await interaction.response.send_message(
            "❌ Nie udało się zapisać werdyktu. Raport pozostaje oczekujący.",
            ephemeral=True,
        )
        return

    embed = interaction.message.embeds[0] if interaction.message.embeds else discord.Embed()
    is_clean = updated["report_status"] == "VERIFIED_CLEAN"
    embed.color = discord.Color.green() if is_clean else discord.Color.red()
    embed.title = (
        f"✅ ZWERYFIKOWANO JAKO CZYSTE — {updated['control_type']}"
        if is_clean
        else "🔴 POTWIERDZONY CHEAT"
    )
    embed.add_field(
        name="Werdykt sędziego",
        value=f"{updated['report_status']} · profil {updated['control_type']}",
        inline=False,
    )
    try:
        await interaction.response.edit_message(embed=embed, view=None)
    except discord.HTTPException:
        LOGGER.exception("Could not update anti-cheat alert after staff review")


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


class AntiCheatBot(commands.Bot):
    async def close(self):
        global database_pool
        if database_pool is not None:
            await database_pool.close()
            database_pool = None
        await super().close()


bot = AntiCheatBot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    global database_pool, _persistent_view_registered
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
        if urlparse(clean_url).path.lower().endswith((".mp4", ".mov")):
            return clean_url
    return None


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
    raw_report = str(_report_value(report, "raw_report_text", "Brak szczegółów raportu."))
    raw_report = raw_report.replace("```", "'''")[:1800]
    numeric_stats = (
        f"**Średni ruch celownika:** {_finite_metric(report, 'avg_speed', 2)}\n"
        f"**Wariancja pędu:** {_finite_metric(report, 'magnitude_variance', 4)}\n"
        f"**Częstotliwość mikro-szarpnięć:** "
        f"{_finite_metric(report, 'jitter_frequency', 2)} Hz\n"
    )
    status_text = (
        "🔴 WYKRYTO ODCHYLENIA OD NORMY"
        if anomalous
        else "🟢 CZYSTO (Brak podejrzanych zachowań)"
    )
    video_link = quote(video_url, safe=":/?&=%#[]!$'*+,;@")
    embed = discord.Embed(
        title=(
            "⚠️ WYKRYTO POTENCJALNĄ ANOMALIĘ RUCHU"
            if anomalous
            else "✅ MATERIAŁ ZWERYFIKOWANY - RUCH W NORMIE"
        ),
        color=discord.Color.orange() if anomalous else discord.Color.green(),
    )
    embed.description = (
        "**Materiały do ręcznej weryfikacji przez sędziego:**\n"
        f"**Plik:** {filename}\n"
        f"**Gracz:** {player_name} (ID: {player_id})\n"
        f"**Wideo:** [Otwórz wideo]({video_link})\n"
        f"**Profil sterowania:** {control_type}\n"
        f"{numeric_stats}\n"
        f"**Status:** {status_text}\n\n"
        "**Logi analizatora wizyjnego:**\n"
        f"```text\n{raw_report}\n```\n"
        "*To sygnał do ręcznej weryfikacji, nie werdykt ani dowód oszustwa.*"
    )
    if len(embed.description) > 4096:
        overflow = len(embed.description) - 4096
        raw_report = raw_report[: max(0, len(raw_report) - overflow - 4)]
        embed.description = (
            embed.description.split("```text\n", 1)[0]
            + "```text\n"
            + raw_report
            + "\n```\n*To sygnał do ręcznej weryfikacji, nie werdykt ani dowód oszustwa.*"
        )
    embed.add_field(
        name="ID raportu",
        value=str(_report_value(report, "id", "n/d"))[:100],
        inline=False,
    )
    embed.set_footer(
        text=f"Raport #{_report_value(report, 'id', 'n/d')} · System asysty sędziowskiej BPL"
    )
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
        return
    if database_pool is None:
        LOGGER.error(
            "Cannot accept video message %s: database connection is not initialized",
            message.id,
        )
        return

    await processing_semaphore.acquire()
    temp_path = None
    report_persisted = False
    needs_dispatch = False
    try:
        async with _lock_source_message(message.id):
            filename = _filename_from_message(message, video_url)
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
                fd, temp_path = tempfile.mkstemp(prefix="verify_", suffix=".mp4")
                os.close(fd)
                await download_video(video_url, temp_path)
                result = await analyze_video_async(
                    SafeBallisticAnalyzer(), temp_path, profiles
                )
                if result.get("powod"):
                    raise RuntimeError(result["powod"])

                async with database_pool.acquire() as connection:
                    report_id = await connection.fetchval(
                        """
                        INSERT INTO reports (
                            source_message_id, guild_id, player_id, player_name, video_url,
                            control_type, avg_speed, magnitude_variance, jitter_frequency,
                            is_anomalous, report_status, raw_report_text
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'PENDING', $11)
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

            confirmation_sent = await _send_submission_notice(
                message,
                f"💚 Dziękujemy za zgłoszenie, {message.author.mention}! "
                f"Twój plik {filename} został zapisany do weryfikacji."
            )
            if not confirmation_sent:
                LOGGER.error(
                    "Could not confirm persisted report %s; keeping source message %s",
                    report_id,
                    message.id,
                )
                return
            try:
                await message.delete()
            except discord.HTTPException:
                LOGGER.exception(
                    "Could not remove source message %s after report %s was persisted",
                    message.id,
                    report_id,
                )
                return

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
                    return
                await _dispatch_report(report_id, alerts_channel)
    except FileTooLargeError:
        LOGGER.warning("Report processing failed: video too large (message %s)", message.id)
        await _send_submission_notice(
            message,
            "Nie udało się przyjąć zgłoszenia: plik przekracza dozwolony limit. "
            "Wiadomość nie została usunięta."
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        LOGGER.exception("Report processing failed for message %s", message.id)
        await _send_submission_notice(
            message,
            "Nie udało się przetworzyć zgłoszenia z powodu błędu pobierania lub pliku. "
            "Wiadomość pozostała na kanale — spróbuj ponownie później."
        )
    except Exception:
        LOGGER.exception("Failed to process or dispatch report for message %s", message.id)
        if report_persisted:
            status_message = (
                "Zgłoszenie zostało zapisane, ale wystąpił problem z wysłaniem "
                "powiadomienia do sędziów. System ponowi wysyłkę; nie wysyłaj "
                "ponownie tego samego zgłoszenia."
            )
        else:
            status_message = (
                "Nie udało się przeanalizować lub zapisać zgłoszenia. "
                "Wiadomość pozostała na kanale — spróbuj ponownie później."
            )
        await _send_submission_notice(message, status_message)
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
