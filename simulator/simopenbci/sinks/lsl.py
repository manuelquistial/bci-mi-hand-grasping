"""LSL StreamOutlet sample sink."""
from __future__ import annotations

from dataset_loaders.base import DatasetMetadata


class LslSampleSink:
    """Publishes samples to Lab Streaming Layer."""

    def __init__(self, source_id: str = "simopenbci-1") -> None:
        self._source_id = source_id
        self._outlet = None
        self._info = None
        self._stream_metadata: DatasetMetadata | None = None
        self._stream_name_used: str | None = None

    @property
    def stream_name(self) -> str | None:
        return self._stream_name_used

    def open(self, metadata: DatasetMetadata, stream_name: str | None = None) -> None:
        from pylsl import StreamInfo, StreamOutlet

        name = (stream_name or metadata.name).strip() or metadata.name
        if (
            self._outlet is not None
            and self._stream_metadata == metadata
            and self._stream_name_used == name
        ):
            return

        self._stream_metadata = metadata
        self._stream_name_used = name
        info = StreamInfo(
            name,
            "EEG",
            metadata.n_channels,
            metadata.sampling_rate,
            "float32",
            self._source_id,
        )
        channels = info.desc().append_child("channels")
        names = metadata.channel_names or [f"Ch{i+1}" for i in range(metadata.n_channels)]
        for ch_name in names[: metadata.n_channels]:
            channels.append_child("channel").append_child_value("label", ch_name)

        self._info = info
        self._outlet = StreamOutlet(info)
        print(f"Stream LSL: {name}, {metadata.n_channels} ch, {metadata.sampling_rate} Hz")

    def push(self, sample: list[float]) -> float:
        """Push one EEG sample; returns the LSL timestamp used."""
        if self._outlet is None:
            raise RuntimeError("LslSampleSink.open() must be called before push()")
        from pylsl import local_clock

        ts = local_clock()
        self._outlet.push_sample(sample, timestamp=ts)
        return ts

    def close(self) -> None:
        self._outlet = None
        self._info = None
        self._stream_metadata = None
        self._stream_name_used = None
