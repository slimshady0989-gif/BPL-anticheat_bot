import asyncio
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
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
bot = commands.Bot(command_prefix="!", intents=intents)

KANAL_ANTYCHEAT_MAGAZYN = {}
KANAL_ANTYCHEAT_ALERTY = {}
ANTICHEAT_CATEGORY_NAME = "🏆 BPL ANTI-CHEAT"
ANTICHEAT_ARCHIVE_CHANNEL = "ac-magazyn-plikow"
ANTICHEAT_ALERT_CHANNEL = "ac-alerty-sedziowskie"
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024

executor = ThreadPoolExecutor(max_workers=2)
processing_semaphore = asyncio.Semaphore(2)


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


class SafeBallisticAnalyzer:
    def __init__(self):
        self.MAX_VELOCITY_NORMAL = 28.2

    def procesuj_wideo_synchronicznie(self, video_path):
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
            podejrzane_momenty = []

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
                    if np.isfinite(position_ms) and position_ms > prev_timestamp
                    else frame_idx / fps
                )
                elapsed = timestamp - prev_timestamp
                if elapsed > max(0.5, 3.0 / fps):
                    LOGGER.warning(
                        "Skipping optical-flow comparison across a %.2fs frame gap in %s",
                        elapsed,
                        video_path,
                    )
                    prev_gray = gray
                    prev_timestamp = timestamp
                    continue

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_speed = float(np.mean(magnitude))

                if avg_speed > self.MAX_VELOCITY_NORMAL:
                    podejrzane_momenty.append(
                        f"• Sekunda {timestamp:.2f}s (Wskaźnik pędu: {avg_speed:.2f})"
                    )
                prev_gray = gray
                prev_timestamp = timestamp

            warto_sprawdzic = len(podejrzane_momenty) > 5
            return {
                "alert": warto_sprawdzic,
                "raport": (
                    "\n".join(podejrzane_momenty[:15])
                    if warto_sprawdzic
                    else "Brak krytycznych odchyleń."
                ),
            }
        except cv2.error:
            LOGGER.warning("OpenCV could not decode video %s", video_path, exc_info=True)
            return {"alert": False, "powod": "Błąd odczytu struktury pliku wideo."}
        finally:
            if cap is not None:
                cap.release()


async def analyze_video_async(analyzer, video_path):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, analyzer.procesuj_wideo_synchronicznie, str(video_path)
    )


def _source_message_id(interaction):
    for embed in interaction.message.embeds:
        for field in embed.fields:
            if field.name == "Źródłowa wiadomość magazynu":
                try:
                    return int(field.value)
                except (TypeError, ValueError):
                    return None
    return None


async def setup_server_channels(guild):
    try:
        category = next(
            (
                item
                for item in guild.categories
                if item.name == ANTICHEAT_CATEGORY_NAME
            ),
            None,
        )
        if category is None:
            category = await guild.create_category(ANTICHEAT_CATEGORY_NAME)
            LOGGER.info(
                "Created anti-cheat category '%s' in guild %s",
                ANTICHEAT_CATEGORY_NAME,
                guild.id,
            )

        bot_member = guild.me
        if bot_member is None:
            LOGGER.error("Cannot provision anti-cheat channels in guild %s: bot member unavailable", guild.id)
            return False

        everyone = guild.default_role
        archive_overwrites = dict(category.overwrites)
        archive_overwrites[everyone] = discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            read_message_history=False,
        )
        archive_overwrites[bot_member] = discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            manage_messages=True,
            attach_files=True,
        )

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
                ANTICHEAT_ARCHIVE_CHANNEL,
                category=category,
                overwrites=archive_overwrites,
            )
            LOGGER.info("Created #%s in guild %s", ANTICHEAT_ARCHIVE_CHANNEL, guild.id)
        elif dict(archive_channel.overwrites) != archive_overwrites:
            await archive_channel.edit(overwrites=archive_overwrites)
            LOGGER.info("Updated permissions for #%s in guild %s", ANTICHEAT_ARCHIVE_CHANNEL, guild.id)

        alert_overwrites = dict(category.overwrites)
        alert_overwrites[everyone] = discord.PermissionOverwrite(
            view_channel=False,
            read_message_history=False,
        )
        alert_overwrites[bot_member] = discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            manage_messages=True,
            attach_files=True,
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
                ANTICHEAT_ALERT_CHANNEL,
                category=category,
                overwrites=alert_overwrites,
            )
            LOGGER.info("Created restricted #%s in guild %s", ANTICHEAT_ALERT_CHANNEL, guild.id)
        elif dict(alert_channel.overwrites) != alert_overwrites:
            await alert_channel.edit(overwrites=alert_overwrites)
            LOGGER.info("Updated permissions for restricted #%s in guild %s", ANTICHEAT_ALERT_CHANNEL, guild.id)

        KANAL_ANTYCHEAT_MAGAZYN[guild.id] = archive_channel.id
        KANAL_ANTYCHEAT_ALERTY[guild.id] = alert_channel.id
        LOGGER.info(
            "Anti-cheat channels ready in guild %s (archive=%s, alerts=%s)",
            guild.id,
            archive_channel.id,
            alert_channel.id,
        )
        return True
    except (discord.Forbidden, discord.HTTPException) as error:
        LOGGER.error(
            "Discord rejected anti-cheat channel provisioning in guild %s: %s",
            guild.id,
            error,
            exc_info=True,
        )
        return False
    except Exception:
        LOGGER.exception("Unexpected error provisioning anti-cheat channels in guild %s", guild.id)
        return False


class PersistentAdminView(discord.ui.View):
    def __init__(self, magazyn_msg_id=None):
        super().__init__(timeout=None)
        self.magazyn_msg_id = magazyn_msg_id

    @discord.ui.button(
        label="Zgłoszone / Wyczyszczone",
        style=discord.ButtonStyle.success,
        custom_id="ac_clear_btn",
        emoji="🗑️",
    )
    async def clear_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        alert_channel_id = KANAL_ANTYCHEAT_ALERTY.get(interaction.guild_id)
        if alert_channel_id is None or interaction.channel_id != alert_channel_id:
            await interaction.response.send_message(
                "❌ Ten przycisk działa wyłącznie na kanale alertów anty-cheat.",
                ephemeral=True,
            )
            return
        if interaction.guild is None or not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ Tylko administrator ligi może zarządzać tym zgłoszeniem.",
                ephemeral=True,
            )
            return

        source_id = self.magazyn_msg_id or _source_message_id(interaction)
        if source_id is not None:
            archive_channel_id = KANAL_ANTYCHEAT_MAGAZYN.get(interaction.guild_id)
            magazyn_channel = bot.get_channel(archive_channel_id) if archive_channel_id else None
            if magazyn_channel is not None:
                try:
                    source_message = await magazyn_channel.fetch_message(source_id)
                    await source_message.delete()
                except discord.NotFound:
                    LOGGER.info("Source message %s was already removed", source_id)
                except discord.HTTPException:
                    LOGGER.exception("Could not delete source message %s", source_id)

        embed = interaction.message.embeds[0]
        embed.color = discord.Color.green()
        embed.title = "✅ MATERIAŁ ZWERYFIKOWANY I ZARCHIWIZOWANY"
        try:
            await interaction.response.edit_message(embed=embed, view=None)
        except discord.HTTPException:
            LOGGER.exception("Could not update anti-cheat alert after staff review")


_persistent_view_registered = False


@bot.event
async def on_ready():
    global _persistent_view_registered
    for guild in bot.guilds:
        await setup_server_channels(guild)
    if not _persistent_view_registered:
        bot.add_view(PersistentAdminView())
        _persistent_view_registered = True
    LOGGER.info("Anty-Cheat BPL is ready for manual review assistance in %s guild(s).", len(bot.guilds))


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


@bot.event
async def on_message(message):
    archive_channel_id = KANAL_ANTYCHEAT_MAGAZYN.get(message.guild.id) if message.guild else None
    if message.author == bot.user or message.channel.id != archive_channel_id:
        return

    video_url = _video_url_from_message(message)
    if video_url is None:
        return
    if processing_semaphore.locked():
        LOGGER.warning("Rejecting video message %s: processing capacity is full", message.id)
        return

    filename = _filename_from_message(message, video_url)
    try:
        await message.delete()
    except discord.HTTPException:
        LOGGER.exception("Could not immediately remove submitted video message %s", message.id)

    try:
        await message.channel.send(
            f"💚 Dziękujemy za zgłoszenie, {message.author.mention}! "
            f"Twój plik {filename} trafił do bezpiecznego magazynu. "
            "Sędziowie właśnie rozpoczęli analizę.",
            delete_after=15,
        )
    except discord.HTTPException:
        LOGGER.exception("Could not send submission confirmation for message %s", message.id)

    async with processing_semaphore:
        fd, temp_path = tempfile.mkstemp(prefix="verify_", suffix=".mp4")
        os.close(fd)
        try:
            await download_video(video_url, temp_path)
            wynik = await analyze_video_async(SafeBallisticAnalyzer(), temp_path)
            alert_channel_id = KANAL_ANTYCHEAT_ALERTY.get(message.guild.id)
            alerty_channel = bot.get_channel(alert_channel_id) if alert_channel_id else None

            if alerty_channel:
                anomalous = wynik["alert"]
                embed = discord.Embed(
                    title=(
                        "⚠️ WYKRYTO POTENCJALNĄ ANOMALIĘ RUCHU"
                        if anomalous
                        else "✅ MATERIAŁ ZWERYFIKOWANY - RUCH W NORMIE"
                    ),
                    color=(
                        discord.Color.orange()
                        if anomalous
                        else discord.Color.green()
                    ),
                )
                embed.description = (
                    "**Materiały do ręcznej weryfikacji przez sędziego:**\n"
                    f"**Plik:** {filename}\n"
                    f"**Gracz:** {message.author.mention} (ID: {message.author.id})\n"
                    f"**Wideo:** [Otwórz wideo]({video_url})\n\n"
                    f"**Status:** "
                    f"{'🔴 WYKRYTO ODCHYLENIA OD NORMY' if anomalous else '🟢 CZYSTO (Brak podejrzanych zachowań)'}\n\n"
                    "**Logi analizatora wizyjnego (Optical Flow peaks):**\n"
                    f"```markdown\n{wynik['raport']}\n```\n\n"
                    "*To sygnał do ręcznej weryfikacji, nie werdykt ani dowód oszustwa.*"
                )
                embed.add_field(
                    name="Źródłowa wiadomość magazynu",
                    value=str(message.id),
                    inline=False,
                )
                embed.set_footer(text="System asysty sędziowskiej BPL")
                try:
                    await alerty_channel.send(embed=embed, view=PersistentAdminView())
                except discord.HTTPException:
                    LOGGER.exception("Could not dispatch report for video message %s", message.id)
            else:
                LOGGER.error(
                    "Cannot dispatch report for video message %s: alert channel %s unavailable",
                    message.id,
                    alert_channel_id,
                )
        except FileTooLargeError:
            LOGGER.warning("Rejected oversized video from message %s", message.id)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            LOGGER.exception("Failed to download or process video from message %s", message.id)
        except (cv2.error, RuntimeError):
            LOGGER.exception("Unexpected video processing failure for message %s", message.id)
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not remove temporary video %s", temp_path)


if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN must be set.")
    bot.run(token)
