"""Discover LSL streams on the local network."""

from __future__ import annotations

from typing import Any

from pipeline.models import StreamSummary


def _channel_labels(info: Any) -> tuple[str, ...]:
    labels: list[str] = []
    try:
        channels = info.desc().child("channels").child("channel")
        while channels.name() == "channel":
            label = channels.child_value("label")
            labels.append(label or f"Ch{len(labels) + 1}")
            channels = channels.next_sibling()
    except Exception:
        return ()
    return tuple(labels)


def summarize_stream(info: Any) -> StreamSummary:
    """Convert a pylsl StreamInfo into a StreamSummary."""
    return StreamSummary(
        name=info.name(),
        stream_type=info.type(),
        channel_count=int(info.channel_count()),
        nominal_srate=float(info.nominal_srate()),
        channel_format=str(info.channel_format()),
        source_id=info.source_id() or "",
        hostname=info.hostname() or "",
        uid=info.uid() or "",
        channel_labels=_channel_labels(info),
    )


def discover_streams(
    *,
    stream_type: str | None = None,
    name: str | None = None,
    wait_seconds: float = 2.0,
) -> list[StreamSummary]:
    """
    Resolve available LSL streams.

    Args:
        stream_type: Optional LSL type filter (e.g. ``"EEG"``).
        name: Optional exact stream name filter.
        wait_seconds: How long to wait for discovery responses.
    """
    from pylsl import resolve_byprop, resolve_streams

    if name:
        infos = resolve_byprop("name", name, minimum=0, timeout=wait_seconds)
    elif stream_type:
        infos = resolve_byprop("type", stream_type, minimum=0, timeout=wait_seconds)
    else:
        infos = resolve_streams(wait_time=wait_seconds)

    summaries = [summarize_stream(info) for info in infos]
    if name:
        summaries = [s for s in summaries if s.name == name]
    if stream_type and name:
        summaries = [s for s in summaries if s.stream_type == stream_type]
    return summaries
