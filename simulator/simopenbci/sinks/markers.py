"""LSL Markers outlet for trial/cue events."""

from __future__ import annotations

from dataset_loaders.events import TrialEvent


class LslMarkerSink:
    """Publishes irregular string markers to Lab Streaming Layer."""

    def __init__(self, source_id: str = "simopenbci-markers") -> None:
        self._source_id = source_id
        self._outlet = None
        self._stream_name: str | None = None

    def open(self, eeg_stream_name: str) -> None:
        from pylsl import StreamInfo, StreamOutlet

        name = f"{eeg_stream_name}_Markers"
        if self._outlet is not None and self._stream_name == name:
            return
        info = StreamInfo(
            name,
            "Markers",
            1,
            0.0,  # irregular rate
            "string",
            self._source_id,
        )
        info.desc().append_child_value("eeg_stream", eeg_stream_name)
        self._outlet = StreamOutlet(info)
        self._stream_name = name
        print(f"Stream LSL Markers: {name}")

    def push(self, event: TrialEvent, timestamp: float | None = None) -> None:
        if self._outlet is None:
            raise RuntimeError("LslMarkerSink.open() must be called before push()")
        if timestamp is None:
            self._outlet.push_sample([event.marker_string])
        else:
            self._outlet.push_sample([event.marker_string], timestamp=timestamp)

    def close(self) -> None:
        self._outlet = None
        self._stream_name = None
