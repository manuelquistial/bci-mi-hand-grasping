"""Minimal EDF reader for continuous EEG (European Data Format)."""

from __future__ import annotations

import array
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def _text(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


@dataclass(frozen=True)
class EdfSignal:
    label: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    samples_per_record: int


@dataclass(frozen=True)
class EdfHeader:
    n_records: int
    record_duration_s: float
    header_bytes: int
    signals: tuple[EdfSignal, ...]

    @property
    def sampling_rate(self) -> float:
        duration = self.record_duration_s
        if duration <= 0:
            raise ValueError("EDF record duration must be positive")
        return self.signals[0].samples_per_record / duration

    @property
    def record_bytes(self) -> int:
        return sum(signal.samples_per_record for signal in self.signals) * 2

    def sample_offset(self, channel: int) -> int:
        return sum(signal.samples_per_record for signal in self.signals[:channel])


def read_header(file_path: str | Path) -> EdfHeader:
    path = Path(file_path)
    with path.open("rb") as handle:
        prefix = handle.read(256)
        if len(prefix) < 256:
            raise ValueError(f"{path.name} is not a valid EDF (header truncated)")
        try:
            n_signals = int(_text(prefix[252:256]))
            header_bytes = int(_text(prefix[184:192]))
            n_records = int(_text(prefix[236:244]))
            record_duration_s = float(_text(prefix[244:252]))
        except ValueError as exc:
            raise ValueError(f"{path.name} has an unreadable EDF header") from exc
        if n_signals <= 0:
            raise ValueError(f"{path.name} declares no EDF signals")
        channel_bytes = n_signals * 256
        channels = handle.read(channel_bytes)
        if len(channels) < channel_bytes:
            raise ValueError(f"{path.name} is not a valid EDF (channel header truncated)")

    def field(offset: int, width: int) -> list[str]:
        return [
            _text(channels[offset + index * width : offset + (index + 1) * width])
            for index in range(n_signals)
        ]

    # Channel header is grouped by field, not by signal:
    # label 16, transducer 80, dimension 8, physical min/max 8, digital min/max 8,
    # prefiltering 80, samples per record 8, reserved 32.
    labels = field(0, 16)
    after_dimension = n_signals * (16 + 80 + 8)
    physical_min = field(after_dimension, 8)
    physical_max = field(after_dimension + n_signals * 8, 8)
    digital_min = field(after_dimension + n_signals * 16, 8)
    digital_max = field(after_dimension + n_signals * 24, 8)
    samples = field(after_dimension + n_signals * 32 + n_signals * 80, 8)
    signals: list[EdfSignal] = []
    for index, label in enumerate(labels):
        try:
            signals.append(
                EdfSignal(
                    label=label,
                    physical_min=float(physical_min[index]),
                    physical_max=float(physical_max[index]),
                    digital_min=int(float(digital_min[index])),
                    digital_max=int(float(digital_max[index])),
                    samples_per_record=int(samples[index]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path.name} signal {index} has an unreadable header") from exc
    return EdfHeader(
        n_records=n_records,
        record_duration_s=record_duration_s,
        header_bytes=header_bytes,
        signals=tuple(signals),
    )


def _scale(signal: EdfSignal) -> tuple[float, float]:
    span = signal.digital_max - signal.digital_min
    if span == 0:
        return 0.0, signal.physical_min
    scale = (signal.physical_max - signal.physical_min) / span
    return scale, signal.physical_min - signal.digital_min * scale


def iter_samples(
    file_path: str | Path,
    header: EdfHeader,
    channels: list[int],
) -> Iterator[list[float]]:
    """Yield one physical sample (µV) per selected channel, in channel order."""
    if not channels:
        return
    scales: list[float] = []
    offsets: list[float] = []
    for channel in channels:
        scale, offset = _scale(header.signals[channel])
        scales.append(scale)
        offsets.append(offset)
    spr = header.signals[channels[0]].samples_per_record
    if any(header.signals[channel].samples_per_record != spr for channel in channels):
        raise ValueError("EDF signals selected for playback must share samples per record")
    starts = [header.sample_offset(channel) for channel in channels]
    record_bytes = header.record_bytes
    limit = header.n_records if header.n_records >= 0 else None
    seen = 0
    with Path(file_path).open("rb") as handle:
        handle.seek(header.header_bytes)
        while limit is None or seen < limit:
            block = handle.read(record_bytes)
            if len(block) < record_bytes:
                break
            raw = array.array("h")
            raw.frombytes(block)
            if sys.byteorder != "little":
                raw.byteswap()
            columns = [raw[start : start + spr] for start in starts]
            for sample_index in range(spr):
                yield [
                    columns[channel][sample_index] * scales[channel] + offsets[channel]
                    for channel in range(len(channels))
                ]
            seen += 1


def iter_channel(
    file_path: str | Path,
    header: EdfHeader,
    channel: int,
) -> Iterator[float]:
    """Yield physical values of one signal, record by record."""
    signal = header.signals[channel]
    scale, offset = _scale(signal)
    start = header.sample_offset(channel)
    count = signal.samples_per_record
    record_bytes = header.record_bytes
    limit = header.n_records if header.n_records >= 0 else None
    seen = 0
    with Path(file_path).open("rb") as handle:
        handle.seek(header.header_bytes)
        while limit is None or seen < limit:
            block = handle.read(record_bytes)
            if len(block) < record_bytes:
                break
            raw = array.array("h")
            raw.frombytes(block[start * 2 : (start + count) * 2])
            if sys.byteorder != "little":
                raw.byteswap()
            for digital in raw:
                yield digital * scale + offset
            seen += 1
