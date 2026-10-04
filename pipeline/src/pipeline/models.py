"""Shared data models for LSL discovery and reading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamSummary:
    """Lightweight view of an LSL stream's public metadata."""

    name: str
    stream_type: str
    channel_count: int
    nominal_srate: float
    channel_format: str
    source_id: str
    hostname: str
    uid: str
    channel_labels: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        labels = ", ".join(self.channel_labels[:8])
        if len(self.channel_labels) > 8:
            labels += ", …"
        extra = f" [{labels}]" if labels else ""
        return (
            f"{self.name} | type={self.stream_type} | "
            f"{self.channel_count} ch @ {self.nominal_srate:g} Hz | "
            f"host={self.hostname} | source_id={self.source_id}{extra}"
        )


@dataclass(frozen=True)
class SampleChunk:
    """One or more samples pulled from an LSL inlet."""

    samples: list[list[float]] | list[list[str]]
    timestamps: list[float]
