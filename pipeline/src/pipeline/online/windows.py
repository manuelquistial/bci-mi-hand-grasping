"""Extract the five overlapping 2 s windows during a 4 s imagery period."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.epochs import ACTIVITY_SECONDS, activity_windows
from pipeline.online.stream import OnlineStream


@dataclass(frozen=True)
class ImageryWindow:
    """One 2 s slice ready for EMG rejection and classification."""

    index: int
    start: float
    end: float
    eeg: np.ndarray  # (samples, channels), causally filtered when stream filters
    eeg_times: np.ndarray
    emg: np.ndarray  # (samples, channels)
    emg_times: np.ndarray
    complete: bool


class WindowManager:
    """Tracks which of the five activity windows are ready as EEG arrives."""

    def __init__(self, stream: OnlineStream) -> None:
        self._stream = stream
        self._onset: float | None = None
        self._ready: list[bool] = []

    def begin(self, onset: float) -> list[tuple[float, float]]:
        """Start a new imagery period at ``onset`` (LSL / stream time)."""
        self._onset = float(onset)
        spans = activity_windows(self._onset)
        self._ready = [False] * len(spans)
        return spans

    def reset(self) -> None:
        self._onset = None
        self._ready = []

    @property
    def onset(self) -> float | None:
        return self._onset

    def spans(self) -> list[tuple[float, float]]:
        if self._onset is None:
            return []
        return activity_windows(self._onset)

    def poll_ready(self) -> list[ImageryWindow]:
        """Return newly completed windows (each emitted once)."""
        if self._onset is None:
            return []
        newest = self._stream.latest_time()
        if newest is None:
            return []
        found: list[ImageryWindow] = []
        for index, (start, end) in enumerate(self.spans()):
            if self._ready[index] or newest < end:
                continue
            window = self._build(index, start, end)
            if not window.complete:
                continue
            self._ready[index] = True
            found.append(window)
        return found

    def all_complete(self) -> bool:
        return bool(self._ready) and all(self._ready)

    def activity_ended(self, now: float | None = None) -> bool:
        if self._onset is None:
            return False
        stamp = self._stream.latest_time() if now is None else float(now)
        if stamp is None:
            return False
        return stamp >= self._onset + ACTIVITY_SECONDS

    def _build(self, index: int, start: float, end: float) -> ImageryWindow:
        eeg, eeg_times = self._stream.slice_eeg(start, end, filtered=True)
        emg, emg_times = self._stream.slice_emg(start, end)
        expected = max(1, int(round(self._stream.sample_rate_hz * (end - start))))
        complete = eeg.shape[0] >= max(8, int(0.8 * expected))
        return ImageryWindow(
            index=index,
            start=start,
            end=end,
            eeg=eeg,
            eeg_times=eeg_times,
            emg=emg,
            emg_times=emg_times,
            complete=complete,
        )
