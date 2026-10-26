"""Full-frame multi-player vision pipeline for the BPL anti-cheat bot.

Everything in this module is CPU-only and synchronous: it is meant to be called
from the analyzer thread pool. All detectors degrade gracefully (missing YOLO
weights -> motion-blob detector, missing Tesseract -> no HUD text, missing
ffmpeg -> no audio sync) so a single missing optional dependency never blocks a
report. Every verdict produced here is a *signal for a human judge*.
"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("bpl.anticheat.vision")


def _env_float(name, default):
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


# ---------------------------------------------------------------------------
# Tunables (all overridable through environment variables)
# ---------------------------------------------------------------------------
ANALYSIS_WIDTH = 640
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH", str(Path(__file__).with_name("models") / "yolov8n.onnx")
)
YOLO_INPUT_SIZE = 640
YOLO_CONFIDENCE = _env_float("YOLO_CONFIDENCE", 0.2)
YOLO_NMS_IOU = 0.45
DETECT_INTERVAL_S = _env_float("DETECT_INTERVAL_S", 0.2)
TRACK_MAX_AGE_S = 0.6
TRACK_MIN_HITS = 2
TIMER_INTERVAL_S = 0.5
KILLFEED_INTERVAL_S = 1.0
TAG_INTERVAL_S = 1.0
MAX_TAG_READS_PER_STEP = 2
NAME_MIN_VOTES = 2

# Locomotion geometry
MIN_MOVE_SPEED_FW_S = _env_float("MIN_MOVE_SPEED_FW_S", 0.08)
SNAP_TOLERANCE_DEG = _env_float("SNAP_TOLERANCE_DEG", 4.0)
SNAP_RATIO_MIN = _env_float("SNAP_RATIO_MIN", 0.80)
SNAP_MIN_SAMPLES = 30
SNAP_WINDOW_S = 2.0
SNAP_MIN_DIRECTION_SHARE = 0.15
ENTITY_JITTER_HZ_MIN = _env_float("ENTITY_JITTER_HZ_MIN", 9.0)
ARTIFACT_WINDOW_S = _env_float("ARTIFACT_WINDOW_S", 1.5)
ARTIFACT_BRIGHTNESS_JUMP = _env_float("ARTIFACT_BRIGHTNESS_JUMP", 28.0)
ARTIFACT_COLOR_COVERAGE_MIN = _env_float("ARTIFACT_COLOR_COVERAGE_MIN", 0.22)

# Ballistic R physics (frame-widths per second, camera compensated)
BALLISTIC_R_ROUND_SECONDS = _env_float("BALLISTIC_R_ROUND_SECONDS", 120.0)
BALLISTIC_R_SPAWN_WINDOW_S = _env_float("BALLISTIC_R_SPAWN_WINDOW_S", 15.0)
BALLISTIC_R_MAX_SPEED_FW_S = _env_float("BALLISTIC_R_MAX_SPEED_FW_S", 1.6)
BALLISTIC_R_SPAWN_RADIUS_FW = _env_float("BALLISTIC_R_SPAWN_RADIUS_FW", 0.6)
TELEPORT_MIN_JUMP_DIAG = _env_float("TELEPORT_MIN_JUMP_DIAG", 0.28)
TELEPORT_MAX_GAP_S = 0.5
TELEPORT_MIN_APPEARANCE = 0.75
IMPULSE_SPIKE_RATIO = _env_float("IMPULSE_SPIKE_RATIO", 6.0)
IMPULSE_MIN_SPIKE_PX = _env_float("IMPULSE_MIN_SPIKE_PX", 8.0)
AIR_TURN_MIN_DEG = 90.0
AIR_TURN_SPEED_KEEP = 0.7
AIR_TURN_MIN_SPEED_FW_S = _env_float("AIR_TURN_MIN_SPEED_FW_S", 0.4)

LABEL_HARDWARE = (
    "⚠️ POTENTIAL HARDWARE EMULATOR (XIM/CRONUS) SIMULATING PAD ON KEYBOARD"
)
LABEL_TELEPORT = "🔴 POSITION/TELEPORT EXPLOIT"
LABEL_AIRSTRAFE = "🔴 AIR-STRAFE / FLY GLITCH MOVEMENT"
LABEL_AIM = "⚠️ ABNORMAL CROSSHAIR SPEED (POV)"

NICKNAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\- ]{1,18}[A-Za-z0-9]")
TIMER_RE = re.compile(r"(\d{1,2})\s*[:.;]\s*(\d{2})")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    confidence: float
    source: str = "yolo"


@dataclass
class Finding:
    kind: str
    label: str
    entity_id: str
    video_second: float
    evidence: str
    confidence: float = 0.5
    clock: str | None = None
    offender: str | None = None


@dataclass
class KillEvent:
    video_second: float
    killer: str
    victim: str
    raw: str


@dataclass
class AvEvent:
    video_second: float
    kind: str
    audio_z: float
    flash_z: float
    shake_z: float


@dataclass
class MultiPlayerResult:
    findings: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    matrix_rows: list = field(default_factory=list)
    timer_readings: list = field(default_factory=list)
    kill_events: list = field(default_factory=list)
    av_events: list = field(default_factory=list)
    impulse_windows: list = field(default_factory=list)
    artifact_windows: list = field(default_factory=list)
    discovered_nicknames: list = field(default_factory=list)
    detector_name: str = "none"
    ocr_available: bool = False
    audio_available: bool = False
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------
def heading_degrees(vx, vy):
    """Screen heading in degrees, 0 = right, 90 = up (screen y grows downward)."""
    return math.degrees(math.atan2(-vy, vx)) % 360.0


def octant_deviation(angle):
    """Distance in degrees from the nearest multiple of 45."""
    remainder = angle % 45.0
    return min(remainder, 45.0 - remainder)


def angle_between(a, b):
    """Smallest absolute angle (degrees) between two 2D vectors."""
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True)
class SnapResult:
    samples: int
    snapped_ratio: float
    distinct_directions: int
    diagonal_share: float


def analyze_octant_snapping(vectors, min_speed, tolerance=SNAP_TOLERANCE_DEG):
    """Measure how strictly heading vectors sit on the 8 WASD directions.

    A uniformly random heading lands within +-tolerance of an octant axis
    ``16 * tolerance / 360`` of the time (about 18% at 4 degrees).
    """
    angles = [
        heading_degrees(vx, vy)
        for vx, vy in vectors
        if math.hypot(vx, vy) >= min_speed
    ]
    if not angles:
        return SnapResult(0, 0.0, 0, 0.0)
    snapped = [a for a in angles if octant_deviation(a) <= tolerance]
    counts = Counter(int(round(a / 45.0)) % 8 for a in snapped)
    total = len(angles)
    directions = sum(
        1 for count in counts.values() if count / total >= SNAP_MIN_DIRECTION_SHARE
    )
    diagonal = sum(count for octant, count in counts.items() if octant % 2 == 1)
    return SnapResult(
        samples=total,
        snapped_ratio=len(snapped) / total,
        distinct_directions=directions,
        diagonal_share=diagonal / total,
    )


def micro_jitter_hz(vectors, duration_s, smoothing=5):
    """Rate of high-frequency sign reversals of the velocity residual."""
    data = np.asarray(vectors, dtype=np.float64)
    if len(data) < smoothing * 2 or duration_s <= 0:
        return 0.0
    kernel = np.ones(smoothing) / smoothing
    reversals = 0
    for axis in range(2):
        series = data[:, axis]
        residual = series - np.convolve(series, kernel, mode="same")
        speed = np.abs(series)
        amplitude = np.maximum(0.5, 0.25 * speed)
        significant = np.abs(residual) > amplitude
        signs = np.sign(residual)
        flips = (signs[1:] * signs[:-1] < 0) & significant[1:] & significant[:-1]
        reversals += int(np.count_nonzero(flips))
    return reversals / 2.0 / duration_s


def _moving_average(data, window):
    data = np.asarray(data, dtype=np.float64)
    if len(data) == 0 or window <= 1:
        return data
    window = min(window, len(data))
    kernel = np.ones(window) / window
    padded = np.pad(data, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(data.shape[1])],
        axis=1,
    )


def format_clock(seconds):
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Text parsing (pure, unit-testable)
# ---------------------------------------------------------------------------
def parse_timer_text(text):
    """Parse an OCR'd round countdown such as ``0:42`` into seconds."""
    if not text:
        return None
    match = TIMER_RE.search(text)
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
        if seconds < 60 and minutes <= 15:
            return minutes * 60 + seconds
        return None
    digits = text.strip()
    if digits.isdigit() and 0 <= int(digits) <= 600:
        return int(digits)
    return None


def clean_nickname(text):
    if not text:
        return None
    for candidate in NICKNAME_RE.findall(text.replace("|", " ")):
        candidate = candidate.strip(" ._-")
        if 3 <= len(candidate) <= 20 and any(c.isalpha() for c in candidate):
            return candidate
    return None


def match_nickname(text, roster, cutoff=0.8):
    """Map OCR text to the closest known nickname (case-insensitive)."""
    cleaned = clean_nickname(text)
    if cleaned is None or not roster:
        return None
    lookup = {name.lower(): name for name in roster if name}
    hit = difflib.get_close_matches(cleaned.lower(), list(lookup), n=1, cutoff=cutoff)
    return lookup[hit[0]] if hit else None


def parse_killfeed(text, video_second, roster=()):
    """Extract ``killer -> victim`` pairs from OCR'd killfeed lines."""
    events = []
    for line in (text or "").splitlines():
        line = line.strip()
        if len(line) < 5:
            continue
        names = []
        if roster:
            for token in re.split(r"\s{2,}|[>»\u2192|]|\s-\s", line):
                matched = match_nickname(token, roster)
                if matched and matched not in names:
                    names.append(matched)
        else:
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{2,19}", line):
                if token not in names:
                    names.append(token)
        if len(names) >= 2:
            events.append(KillEvent(video_second, names[0], names[1], line[:120]))
    return events


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------
def _prepare_for_ocr(image, scale=3):
    if image is None or image.size == 0:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary)) < 127:
        binary = cv2.bitwise_not(binary)
    return cv2.copyMakeBorder(binary, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)


class NullOcrBackend:
    available = False

    def read(self, image, mode):
        return ""


class TesseractBackend:
    """pytesseract wrapper; ``available`` is False when the binary is missing."""

    _CONFIGS = {
        "timer": "--psm 7 -c tessedit_char_whitelist=0123456789:",
        "tag": "--psm 7",
        "killfeed": "--psm 6",
    }

    def __init__(self):
        self.available = False
        self._pytesseract = None
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self.available = True
        except Exception:
            LOGGER.info("Tesseract OCR unavailable; HUD text extraction disabled")

    def read(self, image, mode):
        if not self.available:
            return ""
        prepared = _prepare_for_ocr(image)
        if prepared is None:
            return ""
        try:
            return self._pytesseract.image_to_string(
                prepared, config=self._CONFIGS.get(mode, "--psm 6"), timeout=2
            )
        except Exception:
            LOGGER.debug("OCR call failed", exc_info=True)
            return ""


class HudReader:
    """Reads the countdown timer, killfeed and overhead nametags."""

    TIMER_ROI = (0.44, 0.02, 0.56, 0.08)
    KILLFEED_ROI = (0.0, 0.80, 0.26, 0.93)

    def __init__(self, backend=None, roster=()):
        self.backend = backend if backend is not None else TesseractBackend()
        self.roster = tuple(roster)

    @property
    def available(self):
        return bool(getattr(self.backend, "available", False))

    @staticmethod
    def _crop(frame, roi):
        height, width = frame.shape[:2]
        x0, y0 = int(roi[0] * width), int(roi[1] * height)
        x1, y1 = int(roi[2] * width), int(roi[3] * height)
        return frame[y0:y1, x0:x1]

    def read_timer(self, frame):
        return parse_timer_text(self.backend.read(self._crop(frame, self.TIMER_ROI), "timer"))

    def read_killfeed(self, frame, video_second):
        text = self.backend.read(self._crop(frame, self.KILLFEED_ROI), "killfeed")
        return parse_killfeed(text, video_second, self.roster)

    def read_tag(self, frame, box):
        height, width = frame.shape[:2]
        x, y, w, h = box
        x0 = int(max(0, x - 0.6 * w))
        x1 = int(min(width, x + 1.6 * w))
        y0 = int(max(0, y - 0.45 * h))
        y1 = int(max(0, y + 0.05 * h))
        if x1 - x0 < 12 or y1 - y0 < 6:
            return None
        text = self.backend.read(frame[y0:y1, x0:x1], "tag")
        return match_nickname(text, self.roster) or clean_nickname(text)


class TimerTracker:
    """Keeps a consistent view of the on-screen countdown."""

    def __init__(self):
        self.readings = []
        self._rejected = 0

    def add(self, video_second, clock_seconds):
        if clock_seconds is None:
            return False
        if self.readings:
            last_t, last_clock = self.readings[-1]
            expected = last_clock - (video_second - last_t)
            if abs(expected - clock_seconds) > 3.0:
                self._rejected += 1
                if self._rejected < 3:
                    return False
        self._rejected = 0
        self.readings.append((video_second, clock_seconds))
        return True

    def clock_at(self, video_second):
        if not self.readings:
            return None
        t, clock = min(self.readings, key=lambda item: abs(item[0] - video_second))
        if abs(t - video_second) > 6.0:
            return None
        return clock - (video_second - t)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def decode_yolo_output(output, ratio, pad, frame_size, confidence, nms_iou):
    """Decode YOLOv8 ``(1, 84, N)`` output into person detections."""
    predictions = np.squeeze(np.asarray(output))
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    person_scores = predictions[:, 4]
    keep = person_scores >= confidence
    if not np.any(keep):
        return []
    predictions, person_scores = predictions[keep], person_scores[keep]
    frame_w, frame_h = frame_size
    boxes = []
    for cx, cy, w, h in predictions[:, :4]:
        x = (cx - w / 2 - pad[0]) / ratio
        y = (cy - h / 2 - pad[1]) / ratio
        boxes.append([int(x), int(y), int(w / ratio), int(h / ratio)])
    indices = cv2.dnn.NMSBoxes(boxes, person_scores.tolist(), confidence, nms_iou)
    detections = []
    for index in np.array(indices).flatten():
        x, y, w, h = boxes[int(index)]
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        if w > 4 and h > 8:
            detections.append(Detection(x, y, w, h, float(person_scores[int(index)])))
    return detections


def _is_viewmodel(detection, frame_size):
    """The POV player's own weapon/arms touch the bottom edge near the centre."""
    frame_w, frame_h = frame_size
    center_x = detection.x + detection.w / 2.0
    return (
        detection.y + detection.h >= 0.97 * frame_h
        and detection.y >= 0.45 * frame_h
        and 0.15 * frame_w <= center_x <= 0.85 * frame_w
    )


def _is_hud_zone(detection, frame_size):
    """Portraits, minimap and killfeed icons are never world entities."""
    frame_w, frame_h = frame_size
    center_x = detection.x + detection.w / 2.0
    center_y = detection.y + detection.h / 2.0
    return (
        center_y < 0.10 * frame_h
        or (center_x > 0.84 * frame_w and center_y < 0.40 * frame_h)
        or (center_x < 0.26 * frame_w and center_y > 0.80 * frame_h)
    )


def filter_world_detections(detections, frame_size):
    return [
        d
        for d in detections
        if not _is_viewmodel(d, frame_size) and not _is_hud_zone(d, frame_size)
    ]


class YoloNanoDetector:
    """YOLOv8-Nano (ONNX) person detector running on OpenCV DNN / CPU."""

    name = "yolov8n-onnx"

    def __init__(self, model_path=YOLO_MODEL_PATH, confidence=YOLO_CONFIDENCE):
        self.confidence = confidence
        self.net = None
        path = Path(model_path)
        if path.is_file():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except cv2.error:
                LOGGER.warning("Could not load YOLO model %s", path, exc_info=True)
                self.net = None
        else:
            LOGGER.info("YOLO weights not found at %s", path)

    @property
    def available(self):
        return self.net is not None

    def detect(self, frame, residual=None):
        if self.net is None:
            return []
        height, width = frame.shape[:2]
        size = YOLO_INPUT_SIZE
        ratio = min(size / height, size / width)
        new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
        pad = ((size - new_w) / 2.0, (size - new_h) / 2.0)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        left, top = int(round(pad[0])), int(round(pad[1]))
        canvas[top : top + new_h, left : left + new_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()
        detections = decode_yolo_output(
            output, ratio, (left, top), (width, height), self.confidence, YOLO_NMS_IOU
        )
        return filter_world_detections(detections, (width, height))


class MotionBlobDetector:
    """Fallback silhouette finder: blobs moving differently from the camera."""

    name = "motion-blob-fallback"
    available = True

    def __init__(self, min_area_frac=0.0015, max_area_frac=0.25, threshold_px=1.2):
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.threshold_px = threshold_px

    def detect(self, frame, residual=None):
        if residual is None:
            return []
        height, width = residual.shape[:2]
        magnitude = np.linalg.norm(residual, axis=2)
        mask = (magnitude > self.threshold_px).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(width * height)
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            fraction = w * h / frame_area
            if not self.min_area_frac <= fraction <= self.max_area_frac:
                continue
            if h < 1.1 * w or h < 0.08 * height:
                continue
            detections.append(Detection(x, y, w, h, 0.3, "motion"))
        world = filter_world_detections(detections, (width, height))
        return sorted(world, key=lambda d: d.w * d.h, reverse=True)[:6]


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
def box_iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _appearance(frame, box):
    x, y, w, h = [int(v) for v in box]
    x, y = max(0, x), max(0, y)
    patch = frame[y : y + h, x : x + w]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


@dataclass
class Sample:
    video_second: float
    frame_index: int
    cx: float
    cy: float
    vx: float
    vy: float


@dataclass
class TeleportEvent:
    entity_id: str
    video_second: float
    jump_diag: float
    gap_s: float
    appearance: float


class Track:
    def __init__(self, track_id, detection, frame, cam_cum, video_second):
        self.id = track_id
        self.box = [detection.x, detection.y, detection.w, detection.h]
        self.hits = 1
        self.last_hit_t = video_second
        self.first_t = video_second
        self.hist = _appearance(frame, self.box)
        self.samples = []
        self.name_votes = Counter()
        self.last_tag_t = -1e9
        self.cam_cum = cam_cum
        self.teleport_candidate = None

    @property
    def entity_id(self):
        return f"ENTITY_{self.id}"

    @property
    def confirmed(self):
        return self.hits >= TRACK_MIN_HITS

    @property
    def center(self):
        return self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0

    def name(self):
        if not self.name_votes:
            return None
        name, votes = self.name_votes.most_common(1)[0]
        return name if votes >= NAME_MIN_VOTES else None


class EntityTracker:
    def __init__(self, frame_size, fps):
        self.frame_w, self.frame_h = frame_size
        self.diag = math.hypot(self.frame_w, self.frame_h)
        self.fps = fps
        self.tracks = []
        self.lost = []
        self.all_tracks = []
        self.teleports = []
        self._next_id = 1
        self.cam_cum = np.zeros(2)

    def advance(self, flow_small, cam, video_second, frame_index):
        """Move every live track with the flow field and record a velocity sample."""
        self.cam_cum = self.cam_cum + cam
        for track in self.tracks:
            x, y, w, h = [int(round(v)) for v in track.box]
            x0, y0 = max(0, x + w // 5), max(0, y + h // 5)
            x1, y1 = min(self.frame_w, x + w - w // 5), min(self.frame_h, y + h - h // 5)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            raw = flow_small[y0:y1, x0:x1].reshape(-1, 2).mean(axis=0)
            track.box[0] = float(np.clip(track.box[0] + raw[0], -track.box[2], self.frame_w))
            track.box[1] = float(np.clip(track.box[1] + raw[1], -track.box[3], self.frame_h))
            rel = (raw - cam) * self.fps / self.frame_w
            cx, cy = track.center
            if track.confirmed:
                track.samples.append(
                    Sample(video_second, frame_index, cx, cy, float(rel[0]), float(rel[1]))
                )

    def update(self, detections, frame, video_second):
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                box = (det.x, det.y, det.w, det.h)
                iou = box_iou(track.box, box)
                tcx, tcy = track.center
                dist = math.hypot(tcx - (det.x + det.w / 2), tcy - (det.y + det.h / 2))
                scaled = dist / max(track.box[3], det.h, 1.0)
                if iou >= 0.15 or scaled <= 0.75:
                    pairs.append((-iou, scaled, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            track, det = self.tracks[ti], detections[di]
            track.box = [det.x, det.y, det.w, det.h]
            track.hits += 1
            track.last_hit_t = video_second
            fresh = _appearance(frame, track.box)
            if fresh is not None:
                track.hist = fresh
            if track.teleport_candidate is not None and track.confirmed:
                self.teleports.append(track.teleport_candidate)
                track.teleport_candidate = None

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            track = Track(self._next_id, det, frame, self.cam_cum.copy(), video_second)
            self._next_id += 1
            track.teleport_candidate = self._teleport_for(track, video_second)
            self.tracks.append(track)
            self.all_tracks.append(track)

        survivors = []
        for track in self.tracks:
            if video_second - track.last_hit_t <= TRACK_MAX_AGE_S:
                survivors.append(track)
            elif track.confirmed:
                track.lost_t = video_second
                self.lost.append(track)
        self.tracks = survivors
        self.lost = [t for t in self.lost if video_second - t.lost_t <= TELEPORT_MAX_GAP_S]

    def _teleport_for(self, new_track, video_second):
        if new_track.hist is None:
            return None
        best = None
        for old in self.lost:
            gap = video_second - old.last_hit_t
            if gap > TELEPORT_MAX_GAP_S or old.hist is None:
                continue
            size_ratio = new_track.box[3] / max(old.box[3], 1.0)
            if not 0.75 <= size_ratio <= 1.33:
                continue
            appearance = float(cv2.compareHist(new_track.hist, old.hist, cv2.HISTCMP_CORREL))
            if appearance < TELEPORT_MIN_APPEARANCE:
                continue
            ox, oy = old.center
            nx, ny = new_track.center
            camera_shift = new_track.cam_cum - old.cam_cum
            dx = nx - ox - camera_shift[0]
            dy = ny - oy - camera_shift[1]
            jump = math.hypot(dx, dy) / self.diag
            if jump >= TELEPORT_MIN_JUMP_DIAG and (best is None or jump > best.jump_diag):
                best = TeleportEvent(old.entity_id, video_second, jump, gap, appearance)
        return best


# ---------------------------------------------------------------------------
# Impulse window detection and physics validation
# ---------------------------------------------------------------------------
class ImpulseWindowDetector:
    """Finds shockwave-like camera kicks (frame-wide flow bursts)."""

    def __init__(self, duration_s, fps):
        self.duration_s = duration_s
        self.history = []
        self.windows = []
        self._fps = fps

    def add(self, video_second, global_magnitude):
        self.history.append(global_magnitude)
        span = max(int(self._fps * 2), 10)
        recent = self.history[-span - 1 : -1]
        if len(recent) < span // 2:
            return
        baseline = float(np.median(recent))
        spike = global_magnitude > max(baseline * IMPULSE_SPIKE_RATIO, IMPULSE_MIN_SPIKE_PX)
        if spike and (not self.windows or video_second > self.windows[-1][1]):
            self.windows.append((video_second, video_second + self.duration_s))


def _in_window(t, windows):
    return any(start <= t <= end for start, end in windows)


def _jitter_outside_windows(vectors, times, duration_s, windows):
    """Measure jitter only in contiguous samples outside physics/visual events."""
    if not windows:
        return micro_jitter_hz(vectors, duration_s)
    safe_segments = []
    current = []
    for vector, timestamp in zip(vectors, times):
        if _in_window(float(timestamp), windows):
            if current:
                safe_segments.append(current)
                current = []
        else:
            current.append((vector, float(timestamp)))
    if current:
        safe_segments.append(current)
    observed_duration = 0.0
    weighted_jitter = 0.0
    for segment in safe_segments:
        segment_duration = segment[-1][1] - segment[0][1]
        if len(segment) < 10 or segment_duration <= 0:
            continue
        weighted_jitter += micro_jitter_hz(
            [vector for vector, _ in segment], segment_duration
        ) * segment_duration
        observed_duration += segment_duration
    return weighted_jitter / observed_duration if observed_duration else 0.0


class PhysicsValidator:
    def __init__(self, impulse_multiplier=3.0, timer=None):
        self.impulse_multiplier = impulse_multiplier
        self.timer = timer

    def _clock(self, t):
        return format_clock(self.timer.clock_at(t)) if self.timer else None

    def evaluate(self, tracks, teleports, windows):
        findings = []
        for event in teleports:
            findings.append(
                Finding(
                    kind="TELEPORT",
                    label=LABEL_TELEPORT,
                    entity_id=event.entity_id,
                    video_second=event.video_second,
                    evidence=(
                        f"Entity vanished and re-appeared {event.jump_diag * 100:.0f}% of the "
                        f"frame diagonal away (camera compensated) within {event.gap_s:.2f}s "
                        f"with matching appearance (corr {event.appearance:.2f})."
                    ),
                    confidence=min(0.95, 0.5 + event.jump_diag),
                    clock=self._clock(event.video_second),
                )
            )
        for track in tracks:
            if track.confirmed and track.samples:
                findings.extend(self._speed_findings(track, windows))
                findings.extend(self._air_strafe_findings(track, windows))
        return findings

    def _speed_findings(self, track, windows):
        samples = track.samples
        times = np.array([s.video_second for s in samples])
        speeds = np.array([math.hypot(s.vx, s.vy) for s in samples])
        if len(samples) < 8:
            return []
        smooth = _moving_average(speeds.reshape(-1, 1), 9).ravel()
        for index, speed in enumerate(smooth):
            limit = BALLISTIC_R_MAX_SPEED_FW_S * (
                self.impulse_multiplier if _in_window(times[index], windows) else 1.0
            )
            if speed > limit:
                return [
                    Finding(
                        kind="TELEPORT",
                        label=LABEL_TELEPORT,
                        entity_id=track.entity_id,
                        video_second=float(times[index]),
                        evidence=(
                            f"Camera-compensated speed {speed:.2f} frame-widths/s exceeds the "
                            f"Ballistic R ceiling {limit:.2f} frame-widths/s."
                        ),
                        confidence=0.6,
                        clock=self._clock(float(times[index])),
                    )
                ]
        return self._spawn_budget_findings(track, times, speeds)

    def _spawn_budget_findings(self, track, times, speeds):
        if self.timer is None:
            return []
        start_clock = self.timer.clock_at(float(times[0]))
        if start_clock is None or start_clock < BALLISTIC_R_ROUND_SECONDS - BALLISTIC_R_SPAWN_WINDOW_S:
            return []
        dts = np.diff(times, prepend=times[0])
        travelled = np.cumsum(speeds * dts)
        elapsed = times - times[0]
        budget = BALLISTIC_R_MAX_SPEED_FW_S * elapsed + BALLISTIC_R_SPAWN_RADIUS_FW
        over = np.nonzero(travelled > budget)[0]
        if not len(over):
            return []
        index = int(over[0])
        return [
            Finding(
                kind="TELEPORT",
                label=LABEL_TELEPORT,
                entity_id=track.entity_id,
                video_second=float(times[index]),
                evidence=(
                    f"Entity first seen at match clock {format_clock(start_clock)} (spawn phase) "
                    f"covered {travelled[index]:.2f} frame-widths in {elapsed[index]:.1f}s; "
                    f"the spawn-exit budget allows {budget[index]:.2f}."
                ),
                confidence=0.55,
                clock=self._clock(float(times[index])),
            )
        ]

    def _air_strafe_findings(self, track, windows):
        if not windows:
            return []
        samples = [s for s in track.samples if _in_window(s.video_second, windows)]
        if len(samples) < 8:
            return []
        vectors = _moving_average([(s.vx, s.vy) for s in samples], 5)
        step = max(3, len(samples) // 12)
        for i in range(0, len(samples) - step):
            before, after = vectors[i], vectors[i + step]
            sp_before, sp_after = math.hypot(*before), math.hypot(*after)
            if min(sp_before, sp_after) < AIR_TURN_MIN_SPEED_FW_S:
                continue
            if angle_between(before, after) < AIR_TURN_MIN_DEG:
                continue
            between = min(math.hypot(*v) for v in vectors[i : i + step + 1])
            if between < AIR_TURN_SPEED_KEEP * min(sp_before, sp_after):
                continue
            t = samples[i].video_second
            return [
                Finding(
                    kind="AIR_STRAFE",
                    label=LABEL_AIRSTRAFE,
                    entity_id=track.entity_id,
                    video_second=t,
                    evidence=(
                        f"{angle_between(before, after):.0f} degree direction change during an "
                        f"Impulse window with no speed loss (min {between:.2f} vs "
                        f"{min(sp_before, sp_after):.2f} frame-widths/s) - no wall bounce."
                    ),
                    confidence=0.6,
                    clock=self._clock(t),
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Locomotion (hardware emulator) analysis
# ---------------------------------------------------------------------------
def analyze_locomotion(track, fps, timer=None, impulse_windows=(), artifact_windows=()):
    """Windowed 8-direction snapping + micro-jitter check for one entity."""
    samples = track.samples
    if len(samples) < SNAP_MIN_SAMPLES:
        return None, None
    smooth_n = max(3, int(fps * 0.05))
    vectors = _moving_average([(s.vx, s.vy) for s in samples], smooth_n)
    times = np.array([s.video_second for s in samples])
    best = None
    start = float(times[0])
    while start <= times[-1]:
        mask = (times >= start) & (times < start + SNAP_WINDOW_S)
        window = vectors[mask]
        if len(window) >= SNAP_MIN_SAMPLES:
            snap = analyze_octant_snapping(window, MIN_MOVE_SPEED_FW_S)
            selected_samples = [sample for sample, selected in zip(samples, mask) if selected]
            selected_vectors = [(sample.vx, sample.vy) for sample in selected_samples]
            active_windows = [*impulse_windows, *artifact_windows]
            strict_wasd = (
                snap.samples >= SNAP_MIN_SAMPLES
                and snap.snapped_ratio >= SNAP_RATIO_MIN
                and snap.distinct_directions >= 2
                and snap.diagonal_share >= SNAP_MIN_DIRECTION_SHARE
            )
            jitter = (
                _jitter_outside_windows(
                    selected_vectors,
                    [sample.video_second for sample in selected_samples],
                    SNAP_WINDOW_S,
                    active_windows,
                )
                if strict_wasd
                else micro_jitter_hz(selected_vectors, SNAP_WINDOW_S)
            )
            if (
                snap.samples >= SNAP_MIN_SAMPLES
                and snap.snapped_ratio >= SNAP_RATIO_MIN
                and snap.distinct_directions >= 2
                and snap.diagonal_share >= SNAP_MIN_DIRECTION_SHARE
                and jitter >= ENTITY_JITTER_HZ_MIN
            ):
                if best is None or snap.snapped_ratio > best[1].snapped_ratio:
                    best = (start, snap, jitter)
        start += 1.0
    if best is None:
        return None, None
    start, snap, jitter = best
    clock = format_clock(timer.clock_at(start)) if timer else None
    finding = Finding(
        kind="HARDWARE_EMULATOR",
        label=LABEL_HARDWARE,
        entity_id=track.entity_id,
        video_second=start,
        evidence=(
            f"{snap.snapped_ratio * 100:.0f}% of locomotion vectors sit within "
            f"{SNAP_TOLERANCE_DEG:.0f} degrees of the 8 WASD directions "
            f"({snap.distinct_directions} directions, diagonal share "
            f"{snap.diagonal_share * 100:.0f}%) while aim micro-jitter reaches {jitter:.1f} Hz."
        ),
        confidence=min(0.9, 0.4 + snap.snapped_ratio * 0.5),
        clock=clock,
    )
    return finding, snap


# ---------------------------------------------------------------------------
# Audio / visual sync
# ---------------------------------------------------------------------------
def extract_audio_envelope(video_path, fps, frame_count, timeout=60):
    """Per-frame RMS of the clip audio via ffmpeg, or None when unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or fps <= 0:
        return None
    rate = 16000
    try:
        completed = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(video_path), "-vn", "-ac", "1",
             "-ar", str(rate), "-f", "s16le", "-"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.warning("ffmpeg audio extraction failed", exc_info=True)
        return None
    if completed.returncode != 0 or len(completed.stdout) < rate:
        return None
    audio = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    per_frame = max(int(rate / fps), 1)
    usable = min(frame_count, len(audio) // per_frame)
    if usable <= 0:
        return None
    chunks = audio[: usable * per_frame].reshape(usable, per_frame)
    return np.sqrt(np.mean(chunks**2, axis=1))


def _robust_z(series):
    series = np.asarray(series, dtype=np.float64)
    if len(series) == 0:
        return series
    median = np.median(series)
    mad = np.median(np.abs(series - median)) * 1.4826
    return (series - median) / max(mad, 1e-6)


def find_av_desyncs(audio_env, flash, shake, fps, audio_z_min=6.0):
    """Ballistic audio spikes with neither muzzle flash nor screen shake."""
    if audio_env is None or len(audio_env) < 10 or len(flash) < 10:
        return []
    count = min(len(audio_env), len(flash), len(shake))
    audio_z = _robust_z(np.asarray(audio_env[:count]))
    flash_z = _robust_z(np.abs(np.diff(np.asarray(flash[:count]), prepend=flash[0])))
    shake_z = _robust_z(np.asarray(shake[:count]))
    refractory = max(int(fps * 0.1), 2)
    events, last = [], -refractory
    for i in range(1, count - 1):
        if audio_z[i] < audio_z_min or audio_z[i] < audio_z[i - 1] or audio_z[i] < audio_z[i + 1]:
            continue
        if i - last < refractory:
            continue
        last = i
        lo, hi = max(0, i - 3), min(count, i + 4)
        flash_peak = float(np.max(flash_z[lo:hi]))
        shake_peak = float(np.max(shake_z[max(0, i - 4) : min(count, i + 5)]))
        if flash_peak < 4.0 and shake_peak < 3.5:
            events.append(
                AvEvent(i / fps, "AUDIO_SPIKE_WITHOUT_FLASH_OR_SHAKE",
                        float(audio_z[i]), flash_peak, shake_peak)
            )
    return events


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class MultiPlayerAnalyzer:
    """Per-clip state machine fed one frame (plus optical flow) at a time."""

    def __init__(
        self,
        fps,
        frame_size,
        detector=None,
        hud=None,
        impulse_duration=1.8,
        impulse_multiplier=3.0,
    ):
        native_w, native_h = frame_size
        self.fps = fps
        self.native_size = frame_size
        self.scale = min(1.0, ANALYSIS_WIDTH / native_w)
        self.small_size = (max(int(native_w * self.scale), 16), max(int(native_h * self.scale), 16))
        self.detector = detector if detector is not None else self._default_detector()
        self.hud = hud if hud is not None else HudReader()
        self.tracker = EntityTracker(self.small_size, fps)
        self.timer = TimerTracker()
        self.impulses = ImpulseWindowDetector(impulse_duration, fps)
        self.artifact_windows = []
        self._previous_brightness = None
        self.physics = PhysicsValidator(impulse_multiplier, self.timer)
        self.kill_events = []
        self.flash_series = []
        self.shake_series = []
        self.cam_series = []
        self._last_detect = -1e9
        self._last_timer = -1e9
        self._last_killfeed = -1e9
        self._prev_cam = np.zeros(2)

    @staticmethod
    def _default_detector():
        yolo = YoloNanoDetector()
        return yolo if yolo.available else MotionBlobDetector()

    def process_frame(self, frame, native_flow, video_second, frame_index):
        small = cv2.resize(frame, self.small_size, interpolation=cv2.INTER_AREA)
        flow_small = cv2.resize(native_flow, self.small_size, interpolation=cv2.INTER_AREA) * self.scale
        cam = np.median(flow_small[::8, ::8].reshape(-1, 2), axis=0)
        self.cam_series.append((video_second, float(cam[0]), float(cam[1])))
        self.shake_series.append(float(np.linalg.norm(cam - self._prev_cam)))
        self._prev_cam = cam
        height, width = small.shape[:2]
        roi = small[int(height * 0.3) : int(height * 0.7), int(width * 0.35) : int(width * 0.75)]
        self.flash_series.append(float(np.mean(roi)) if roi.size else 0.0)
        brightness = float(np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array((18, 100, 100)), np.array((38, 255, 255)))
        violet = cv2.inRange(hsv, np.array((125, 70, 70)), np.array((165, 255, 255)))
        color_coverage = float(np.count_nonzero(yellow | violet)) / max(height * width, 1)
        brightness_jump = (
            self._previous_brightness is not None
            and abs(brightness - self._previous_brightness) >= ARTIFACT_BRIGHTNESS_JUMP
        )
        self._previous_brightness = brightness
        if brightness_jump or color_coverage >= ARTIFACT_COLOR_COVERAGE_MIN:
            end = video_second + ARTIFACT_WINDOW_S
            if self.artifact_windows and video_second <= self.artifact_windows[-1][1]:
                self.artifact_windows[-1] = (self.artifact_windows[-1][0], end)
            else:
                self.artifact_windows.append((video_second, end))
        self.impulses.add(video_second, float(np.mean(np.linalg.norm(flow_small[::4, ::4], axis=2))))

        self.tracker.advance(flow_small, cam, video_second, frame_index)

        if video_second - self._last_detect >= DETECT_INTERVAL_S:
            self._last_detect = video_second
            residual = flow_small - cam if isinstance(self.detector, MotionBlobDetector) else None
            detections = self.detector.detect(small, residual)
            self.tracker.update(detections, small, video_second)
            self._read_tags(frame, video_second)

        if self.hud.available:
            if video_second - self._last_timer >= TIMER_INTERVAL_S:
                self._last_timer = video_second
                self.timer.add(video_second, self.hud.read_timer(frame))
            if video_second - self._last_killfeed >= KILLFEED_INTERVAL_S:
                self._last_killfeed = video_second
                self.kill_events.extend(self.hud.read_killfeed(frame, video_second))

    def _read_tags(self, frame, video_second):
        if not self.hud.available:
            return
        inverse = 1.0 / self.scale
        reads = 0
        for track in self.tracker.tracks:
            if reads >= MAX_TAG_READS_PER_STEP:
                break
            if not track.confirmed or video_second - track.last_tag_t < TAG_INTERVAL_S:
                continue
            track.last_tag_t = video_second
            reads += 1
            box = [v * inverse for v in track.box]
            name = self.hud.read_tag(frame, box)
            if name:
                track.name_votes[name] += 1

    def _offender_name(self, finding, track):
        name = track.name() if track else None
        if name:
            return name
        nearby = [e for e in self.kill_events if abs(e.video_second - finding.video_second) <= 3.0]
        if nearby:
            return max(nearby, key=lambda e: -abs(e.video_second - finding.video_second)).killer
        return None

    def finalize(self, jitter_hz_pov=0.0):
        tracks = [t for t in self.tracker.all_tracks if t.confirmed and t.samples]
        by_id = {t.entity_id: t for t in tracks}
        findings = []
        entities = []
        for track in tracks:
            finding, snap = analyze_locomotion(
                track, self.fps, self.timer,
                self.impulses.windows, self.artifact_windows
            )
            if finding:
                findings.append(finding)
            speeds = np.array([math.hypot(s.vx, s.vy) for s in track.samples])
            vectors = [(s.vx, s.vy) for s in track.samples]
            duration = max(track.samples[-1].video_second - track.samples[0].video_second, 1e-6)
            snap_all = analyze_octant_snapping(vectors, MIN_MOVE_SPEED_FW_S)
            strict_wasd = (
                snap_all.samples >= SNAP_MIN_SAMPLES
                and snap_all.snapped_ratio >= SNAP_RATIO_MIN
                and snap_all.distinct_directions >= 2
                and snap_all.diagonal_share >= SNAP_MIN_DIRECTION_SHARE
            )
            entity_jitter = (
                _jitter_outside_windows(
                    vectors,
                    [sample.video_second for sample in track.samples],
                    duration,
                    [*self.impulses.windows, *self.artifact_windows],
                )
                if strict_wasd
                else micro_jitter_hz(vectors, duration)
            )
            entities.append(
                {
                    "entity_id": track.entity_id,
                    "name": track.name(),
                    "samples": len(speeds),
                    "avg_speed": float(np.mean(speeds)),
                    "speed_variance": float(np.var(speeds)),
                    "max_speed": float(np.max(speeds)),
                    "snapped_ratio": snap_all.snapped_ratio,
                    "jitter_hz": entity_jitter,
                    "first_second": track.samples[0].video_second,
                    "last_second": track.samples[-1].video_second,
                }
            )
        findings.extend(
            self.physics.evaluate(tracks, self.tracker.teleports, self.impulses.windows)
        )
        findings = [f for f in findings if f.entity_id in by_id or f.kind == "TELEPORT"]
        notes = []
        if isinstance(self.detector, MotionBlobDetector):
            # Motion blobs are too noisy to accuse anyone; keep them for the log only.
            if findings:
                notes.append(
                    f"{len(findings)} candidate finding(s) suppressed: YOLO weights "
                    "unavailable, fallback motion detector is not reliable enough."
                )
            findings = []
        for finding in findings:
            finding.offender = self._offender_name(finding, by_id.get(finding.entity_id))
        findings.sort(key=lambda f: f.video_second)

        rows = [
            (s.video_second, s.frame_index, t.entity_id, s.cx, s.cy, s.vx, s.vy)
            for t in tracks
            for s in t.samples
        ]
        rows.sort()
        small_w = float(self.small_size[0])
        for index, (t, cam_x, cam_y) in enumerate(self.cam_series):
            rows.append(
                (t, index + 1, "POV_SCENE", small_w / 2, self.small_size[1] / 2,
                 cam_x * self.fps / small_w, cam_y * self.fps / small_w)
            )
        rows.sort(key=lambda row: (row[0], row[2]))
        discovered = []
        for track in self.tracker.all_tracks:
            for name, votes in track.name_votes.items():
                if votes >= 3 and name not in discovered:
                    discovered.append(name)
        for event in self.kill_events:
            for name in (event.killer, event.victim):
                if name not in discovered and clean_nickname(name):
                    discovered.append(name)

        return MultiPlayerResult(
            findings=findings,
            entities=entities,
            matrix_rows=rows,
            timer_readings=list(self.timer.readings),
            kill_events=list(self.kill_events),
            impulse_windows=list(self.impulses.windows),
            artifact_windows=list(self.artifact_windows),
            discovered_nicknames=discovered[:12],
            detector_name=getattr(self.detector, "name", "custom"),
            ocr_available=self.hud.available,
            notes=notes,
        )

    def audio_visual_events(self, video_path, frame_count):
        envelope = extract_audio_envelope(video_path, self.fps, frame_count)
        if envelope is None:
            return [], False
        return find_av_desyncs(envelope, self.flash_series, self.shake_series, self.fps), True
