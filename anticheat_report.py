"""Player-profile maths and the detailed analysis log for judges / Epic Support."""

from __future__ import annotations

import math
from datetime import datetime, timezone

MIN_PROFILE_SAMPLES = 5
LAG_PRONE_RATIO = 0.25
MAX_LOG_CHARS = 1_500_000
MAX_MATRIX_ROWS = 4000
REPORT_ID_TOKEN = "{{REPORT_ID}}"
CALIBRATION_ROUND = 1


# ---------------------------------------------------------------------------
# Adaptive per-player thresholds
# ---------------------------------------------------------------------------
def _field(profile, key, default=0.0):
    try:
        value = profile[key]
    except (KeyError, TypeError):
        return default
    return default if value is None else value


def lag_ratio(profile):
    count = int(_field(profile, "sample_count", 0))
    return float(_field(profile, "lag_clip_count", 0)) / count if count else 0.0


def calibration_statistics(baseline, value):
    """Return a baseline-relative z-score and one-sided normal tail probability."""
    if not baseline or int(_field(baseline, "league_round", 0)) != CALIBRATION_ROUND:
        return None, None
    count = int(_field(baseline, "sample_count", 0))
    if count < 2:
        return None, None
    mean = float(_field(baseline, "base_avg_speed", 0.0))
    variance = max(float(_field(baseline, "speed_m2", 0.0)) / (count - 1), 0.0)
    sigma = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(sigma) or sigma <= 0:
        return None, None
    z_score = (float(value) - mean) / sigma
    probability = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return z_score, probability


def personalized_threshold(profile, fallback_threshold, calibration_baseline=None):
    """Player-specific mean + 3 sigma, blended with the control-type fallback.

    Returns ``(threshold, source)``. Until ``MIN_PROFILE_SAMPLES`` judge-verified
    clean clips exist the control-profile threshold is used unchanged. Players
    with a history of desync/lag get a proportionally wider threshold.
    """
    baseline_version = None
    if calibration_baseline:
        if int(_field(calibration_baseline, "league_round", 0)) != CALIBRATION_ROUND:
            raise ValueError("Anti-cheat calibration must use Kolejka 1 data.")
        baseline_count = int(_field(calibration_baseline, "sample_count", 0))
        if baseline_count >= 2:
            baseline_m2 = max(float(_field(calibration_baseline, "speed_m2", 0.0)), 0.0)
            baseline_sigma = math.sqrt(baseline_m2 / (baseline_count - 1))
            fallback_threshold = (
                float(_field(calibration_baseline, "base_avg_speed"))
                + 3.0 * baseline_sigma
            )
            baseline_version = _field(calibration_baseline, "calibration_version", None)
    count = int(_field(profile, "sample_count", 0)) if profile else 0
    if count < MIN_PROFILE_SAMPLES:
        source = f"kolejka-1:{baseline_version}" if baseline_version else "control-profile"
        return float(fallback_threshold), source
    mean = float(_field(profile, "speed_mean"))
    sigma = math.sqrt(max(float(_field(profile, "speed_m2")), 0.0) / (count - 1))
    sigma = max(sigma, 0.1 * abs(mean))
    personal = mean + 3.0 * sigma
    ratio = lag_ratio(profile)
    if ratio >= LAG_PRONE_RATIO:
        personal *= 1.0 + min(ratio, 1.0)
    personal = max(personal, 0.5 * fallback_threshold)
    weight = min(1.0, count / (3.0 * MIN_PROFILE_SAMPLES))
    blended = weight * personal + (1.0 - weight) * fallback_threshold
    source = f"player-profile (n={count}, w={weight:.2f})"
    if baseline_version:
        source = f"kolejka-1:{baseline_version} + {source}"
    return float(blended), source


def adaptive_min_run(profile, base_frames):
    """Desync-prone players must sustain an anomaly longer before it is flagged."""
    count = int(_field(profile, "sample_count", 0)) if profile else 0
    if count >= MIN_PROFILE_SAMPLES and lag_ratio(profile) >= LAG_PRONE_RATIO:
        return base_frames + 2
    return base_frames


# ---------------------------------------------------------------------------
# Headline selection
# ---------------------------------------------------------------------------
def headline_finding(result):
    """Strongest multi-player finding, or None."""
    multi = result.get("multi")
    findings = list(getattr(multi, "findings", []) or [])
    if not findings:
        return None
    priority = {"HARDWARE_EMULATOR": 0, "TELEPORT": 1, "AIR_STRAFE": 1}
    return min(findings, key=lambda f: (priority.get(f.kind, 2), -f.confidence, f.video_second))


def summarize_anomaly(result):
    """Return ``(offender, anomaly_type, second, clock)`` for DB + embed."""
    finding = headline_finding(result)
    if finding is not None:
        offender = finding.offender or f"Nieustalony ({finding.entity_id})"
        return offender, finding.label, float(finding.video_second), finding.clock
    if result.get("alert"):
        return (
            "POV (kamera zgłaszającego)",
            "⚠️ ABNORMAL CROSSHAIR SPEED (POV)",
            result.get("anomaly_second"),
            result.get("anomaly_clock"),
        )
    return None, None, None, None


# ---------------------------------------------------------------------------
# Detailed log
# ---------------------------------------------------------------------------
def _num(value, precision=3):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{precision}f}" if math.isfinite(number) else "n/a"


def _heading(vx, vy):
    if math.hypot(vx, vy) < 1e-9:
        return "  -  "
    return f"{math.degrees(math.atan2(-vy, vx)) % 360.0:5.1f}"


def build_detailed_analysis(result, meta):
    """Plain-text analysis; the report id is substituted at attachment time."""
    multi = result.get("multi")
    offender, anomaly, second, clock = summarize_anomaly(result)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 78,
        f"BALLISTIC POLAND LEAGUE - ANTI-CHEAT DETAILED ANALYSIS (REPORT {REPORT_ID_TOKEN})",
        "=" * 78,
        f"Generated         : {now}",
        f"File              : {meta.get('filename', 'n/a')}",
        f"Submitting player : {meta.get('player_name', 'n/a')} "
        f"(Discord ID {meta.get('player_id', 'n/a')}) - POV / victim",
        f"Detected offender : {offender or 'none'}",
        f"Anomaly type      : {anomaly or 'none'}",
        f"Anomaly second    : {_num(second, 2) if second is not None else 'n/a'} s (video)"
        + (f" | match clock {clock}" if clock else ""),
        "Map               : Ballistic R (zoramaps)",
        f"Detector          : {getattr(multi, 'detector_name', 'n/a')} | OCR: "
        f"{'on' if getattr(multi, 'ocr_available', False) else 'off'} | audio: "
        f"{'on' if getattr(multi, 'audio_available', False) else 'off'}",
        "",
        "1. AUTOMATED VERDICT (decision support - NOT proof; requires judge review)",
        "-" * 78,
        str(result.get("raport", "n/a")),
        f"POV control type guess : {result.get('control_type', 'n/a')}",
        f"POV avg crosshair speed: {_num(result.get('avg_speed'))} | variance "
        f"{_num(result.get('magnitude_variance'), 4)} | micro-jitter "
        f"{_num(result.get('jitter_frequency'), 2)} Hz",
        f"Threshold used         : {_num(result.get('threshold'))} "
        f"[{result.get('threshold_source', 'control-profile')}]",
        f"Calibration            : {result.get('calibration_version', 'legacy-default')}",
        f"Baseline z-score / tail probability: "
        f"{_num(result.get('baseline_z_score'), 2)} / "
        f"{_num(result.get('baseline_probability'), 4)}",
        f"Longest over-threshold run: {result.get('longest_anomaly_run', 0)} frames | "
        f"lag/desync suspected: {bool(result.get('lag_detected'))}",
        "",
        "2. FINDINGS",
        "-" * 78,
    ]
    findings = list(getattr(multi, "findings", []) or [])
    if not findings:
        lines.append("No multi-player locomotion/physics findings.")
    for index, finding in enumerate(findings, 1):
        lines.extend(
            [
                f"[{index}] {finding.label}",
                f"    entity    : {finding.entity_id} ({finding.offender or 'name not resolved'})",
                f"    time      : {_num(finding.video_second, 2)} s video"
                + (f" | match clock {finding.clock}" if finding.clock else ""),
                f"    confidence: {finding.confidence:.2f}",
                f"    evidence  : {finding.evidence}",
            ]
        )

    lines.extend(["", "3. TRACKED ENTITIES (units: frame-widths per second)", "-" * 78])
    entities = list(getattr(multi, "entities", []) or [])
    if entities:
        lines.append(
            "entity       name                 samples  avg_v   var_v   max_v   snap%  jitterHz  seen[s]"
        )
    for entity in entities:
        lines.append(
            f"{entity['entity_id']:<12} {str(entity['name'] or '?'):<20} {entity['samples']:>7} "
            f"{entity['avg_speed']:>6.3f} {entity['speed_variance']:>7.4f} {entity['max_speed']:>7.3f} "
            f"{entity['snapped_ratio'] * 100:>6.1f} {entity['jitter_hz']:>9.1f}  "
            f"{entity['first_second']:.1f}-{entity['last_second']:.1f}"
        )
    if not entities:
        lines.append("No confirmed entities.")

    lines.extend(["", "4. HUD READINGS (OCR)", "-" * 78])
    readings = list(getattr(multi, "timer_readings", []) or [])
    if readings:
        lines.append(
            f"Round timer: {len(readings)} readings, first {readings[0][1]}s left @ "
            f"{readings[0][0]:.1f}s video, last {readings[-1][1]}s left @ {readings[-1][0]:.1f}s video"
        )
    else:
        lines.append("Round timer: no readable countdown (OCR unavailable or HUD hidden).")
    for event in list(getattr(multi, "kill_events", []) or [])[:50]:
        lines.append(f"Killfeed @ {event.video_second:6.1f}s: {event.killer} -> {event.victim}")
    discovered = list(getattr(multi, "discovered_nicknames", []) or [])
    lines.append("Nicknames seen on screen: " + (", ".join(discovered) if discovered else "none"))

    lines.extend(["", "5. IMPULSE / SHOCKWAVE WINDOWS", "-" * 78])
    windows = list(getattr(multi, "impulse_windows", []) or [])
    if windows:
        lines.extend(f"{a:.2f}s - {b:.2f}s" for a, b in windows)
    else:
        lines.append("none detected")

    lines.extend(["", "6. AUDIO-VISUAL SYNC", "-" * 78])
    if not getattr(multi, "audio_available", False):
        lines.append("Audio track unavailable (ffmpeg missing or clip has no audio).")
    else:
        events = list(getattr(multi, "av_events", []) or [])
        lines.append(f"{len(events)} ballistic audio spikes without muzzle flash or screen shake")
        for event in events[:100]:
            lines.append(
                f"  {event.video_second:7.2f}s {event.kind} audio_z={event.audio_z:.1f} "
                f"flash_z={event.flash_z:.1f} shake_z={event.shake_z:.1f}"
            )

    lines.extend(
        ["", "7. FRAME-BY-FRAME VELOCITY MATRIX (frame-widths/s, camera compensated)", "-" * 78]
    )
    rows = list(getattr(multi, "matrix_rows", []) or [])
    stride = max(1, math.ceil(len(rows) / MAX_MATRIX_ROWS))
    if stride > 1:
        lines.append(f"(every {stride}. row shown, {len(rows)} rows total)")
    lines.append("time[s]  frame  entity       cx      cy      vx       vy      speed   heading")
    for t, frame, entity, cx, cy, vx, vy in rows[::stride]:
        lines.append(
            f"{t:7.3f} {frame:6d}  {entity:<11} {cx:7.1f} {cy:7.1f} {vx:8.3f} {vy:8.3f} "
            f"{math.hypot(vx, vy):7.3f}   {_heading(vx, vy)}"
        )

    lines.extend(["", "8. EPIC GAMES SUPPORT TICKET TEXT", "-" * 78])
    lines.append(
        "Automated analysis of a Ballistic Poland League match recording "
        f"({meta.get('filename', 'n/a')}) flagged "
        f"{offender or 'no specific player'} for: {anomaly or 'no anomaly'}"
        + (f" at {_num(second, 2)} s of the recording" if second is not None else "")
        + (f" (round clock {clock})" if clock else "")
        + ". Supporting metrics are listed above. These are statistical indicators "
        "from optical-flow analysis and require manual review of the original footage."
    )
    text = "\n".join(lines) + "\n"
    if len(text) > MAX_LOG_CHARS:
        text = text[:MAX_LOG_CHARS] + "\n[log truncated]\n"
    return text
