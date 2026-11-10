"""Reject imagery windows contaminated by overt muscle activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.online.windows import ImageryWindow


@dataclass(frozen=True)
class EmgRejectConfig:
    """Amplitude gate on a subset of EMG channels.

    The forearm montage has 8 channels; online control uses the first
    ``n_channels`` (default 2) unless ``channel_indices`` is set.
    """

    threshold: float = 50.0
    n_channels: int = 2
    channel_indices: tuple[int, ...] | None = None
    metric: str = "rms"  # "rms" | "peak"


@dataclass(frozen=True)
class EmgVerdict:
    rejected: bool
    amplitude: float
    threshold: float
    reason: str


class EMGRejector:
    """Flag a window when EMG amplitude exceeds the configured threshold."""

    def __init__(self, config: EmgRejectConfig | None = None) -> None:
        self.config = config or EmgRejectConfig()

    def check(self, window: ImageryWindow) -> EmgVerdict:
        return self.check_samples(window.emg)

    def check_samples(self, emg: np.ndarray) -> EmgVerdict:
        array = np.asarray(emg, dtype=np.float64)
        if array.size == 0:
            return EmgVerdict(
                rejected=False,
                amplitude=0.0,
                threshold=self.config.threshold,
                reason="no_emg",
            )
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        columns = self._columns(array.shape[1])
        selected = array[:, columns]
        amplitude = self._amplitude(selected)
        rejected = bool(amplitude > self.config.threshold)
        return EmgVerdict(
            rejected=rejected,
            amplitude=float(amplitude),
            threshold=self.config.threshold,
            reason="emg_contaminated" if rejected else "ok",
        )

    def _columns(self, width: int) -> list[int]:
        if self.config.channel_indices is not None:
            chosen = [int(index) for index in self.config.channel_indices]
        else:
            count = min(self.config.n_channels, width)
            chosen = list(range(count))
        if not chosen:
            raise ValueError("EMG rejector needs at least one channel")
        if max(chosen) >= width:
            raise ValueError("EMG channel index out of range")
        return chosen

    def _amplitude(self, selected: np.ndarray) -> float:
        if self.config.metric == "peak":
            return float(np.max(np.abs(selected)))
        if self.config.metric == "rms":
            return float(np.sqrt(np.mean(selected**2)))
        raise ValueError("EMG metric must be 'rms' or 'peak'")
