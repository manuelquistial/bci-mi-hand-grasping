"""Clock adapters for real-time and instant (test) playback."""
from __future__ import annotations

import time


class RealtimeClock:
    """Wall-clock compensated timing: sleep until t0 + (n+1)/fs."""

    def __init__(self) -> None:
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def wait_for_sample(self, sample_index: int, sampling_rate: float) -> None:
        if self._t0 is None:
            self.start()
        if sampling_rate <= 0:
            return
        # After pushing sample_index (0-based), wait until time for next sample.
        target = self._t0 + (sample_index + 1) / sampling_rate
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


class InstantClock:
    """No-op clock for unit tests."""

    def start(self) -> None:
        return None

    def wait_for_sample(self, sample_index: int, sampling_rate: float) -> None:
        return None
