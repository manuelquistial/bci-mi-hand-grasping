"""Pure helpers for the LSL visualizer: channel parsing, markers, colors."""

from __future__ import annotations

from typing import Sequence

from pipeline.models import StreamSummary
from pipeline.reader import LslStreamReader


def _channel_names(summary: StreamSummary, indices: Sequence[int]) -> list[str]:
    labels = list(summary.channel_labels)
    names: list[str] = []
    for i in indices:
        if i < len(labels) and labels[i]:
            names.append(labels[i])
        else:
            names.append(f"Ch{i + 1}")
    return names


def _parse_channels(spec: str | None, n_channels: int) -> list[int]:
    if not spec:
        return list(range(n_channels))
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    for i in indices:
        if i < 0 or i >= n_channels:
            raise ValueError(f"channel index {i} out of range [0, {n_channels - 1}]")
    if not indices:
        raise ValueError("no channels selected")
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def _curve_color(index: int, n: int) -> tuple[int, int, int]:
    hue = (index / max(1, n)) % 1.0
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _parse_marker_payload(raw: str) -> tuple[str, str]:
    """
    Parse ``kind:code:label`` (simulator convention) into (kind, display_label).
    Falls back to treating the whole string as the label.
    """
    text = (raw or "").strip()
    if not text:
        return "event", "?"
    parts = text.split(":")
    if len(parts) >= 3:
        kind, _code, label = parts[0], parts[1], ":".join(parts[2:])
        return kind or "event", label or _code or text
    if len(parts) == 2:
        return parts[0] or "event", parts[1] or text
    return "event", text


def _segment_color(kind: str, label: str) -> tuple[int, int, int]:
    """Color for a labeled epoch so rest, left, and right stay distinct."""
    key = (label or "").strip().lower()
    if key in {"left", "t1"}:
        return (72, 190, 120)
    if key in {"right", "t2"}:
        return (88, 164, 255)
    if kind == "rest" or key in {"rest", "t0"}:
        return (186, 186, 186)
    if kind == "trial_start":
        return (255, 196, 64)
    return (120, 200, 255)


def _segments_in_window(
    markers: Sequence[tuple[float, str, str]],
    *,
    last_eeg_ts: float,
    window_seconds: float,
) -> list[tuple[float, float, float, str, str, bool]]:
    """Epochs that overlap the visible window.

    Each marker opens a segment that lasts until the next marker. The latest
    segment stays open up to the newest EEG sample (plot x = 0).

    Returns ``(fill_x0, fill_x1, boundary_x, kind, label, boundary_visible)``.
    Coordinates are seconds relative to the newest sample, clipped to the window.
    """
    ordered = sorted(markers, key=lambda item: item[0])
    left = -float(window_seconds)
    right = 0.0
    segments: list[tuple[float, float, float, str, str, bool]] = []
    for index, (timestamp, kind, label) in enumerate(ordered):
        boundary = float(timestamp) - last_eeg_ts
        if index + 1 < len(ordered):
            end = float(ordered[index + 1][0]) - last_eeg_ts
        else:
            end = right
        if end < left or boundary > right:
            continue
        fill_x0 = max(boundary, left)
        fill_x1 = min(end, right)
        if fill_x1 <= fill_x0:
            continue
        segments.append(
            (fill_x0, fill_x1, boundary, kind, label, left <= boundary <= right)
        )
    return segments


def _open_marker_reader(
    *,
    eeg_name: str,
    markers_name: str | None,
    wait_seconds: float,
) -> LslStreamReader | None:
    candidates = []
    if markers_name:
        candidates.append(markers_name)
    candidates.append(f"{eeg_name}_Markers")
    last_error: Exception | None = None
    for name in candidates:
        try:
            return LslStreamReader.open(name=name, wait_seconds=min(wait_seconds, 2.0))
        except TimeoutError as exc:
            last_error = exc
            continue
    # Optional: discover any Markers stream
    try:
        return LslStreamReader.open(stream_type="Markers", wait_seconds=min(wait_seconds, 1.5))
    except TimeoutError:
        if last_error:
            print(f"No Markers stream found (optional). Last error: {last_error}")
        else:
            print("No Markers stream found (optional).")
        return None
