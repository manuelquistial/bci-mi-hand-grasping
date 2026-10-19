"""Fit a descriptor on training trials and apply it to any trial from that fold."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline.features.csp import log_variances, spatial_filters
from pipeline.features.fbcsp import _bandpass, _bands_for, _most_informative
from pipeline.features.psd import band_powers
from pipeline.features.tdp import _keep_channels, _trial_matrix
from pipeline.features.ts import downsample
from pipeline.params import ModelParams


class FeatureEncoder:
    """Supervised descriptors remember only what ``fit`` saw."""

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        raise NotImplementedError

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class TimeSeriesEncoder(FeatureEncoder):
    def __init__(self, points: int = 32) -> None:
        self._points = points

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        return None

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        frames = [downsample(trial, self._points).ravel() for trial in trials]
        length = min(int(frame.size) for frame in frames)
        return np.stack([frame[:length] for frame in frames], axis=0)


class PowerEncoder(FeatureEncoder):
    def __init__(
        self,
        sample_rate_hz: float,
        channels: Sequence[str],
        bands: Sequence[tuple[str, float, float]] | None = None,
    ) -> None:
        self._rate = float(sample_rate_hz)
        self._channels = list(channels)
        self._bands = bands

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        return None

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        _names, values = band_powers(trials, self._rate, self._channels, self._bands)
        return values


class TimeDomainEncoder(FeatureEncoder):
    def __init__(self, channels: Sequence[str]) -> None:
        self._channels = list(channels)
        self._kept: list[int] = []

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        matrices = [_trial_matrix(trial, self._channels) for trial in trials]
        self._kept = _keep_channels(matrices, labels)

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for trial in trials:
            matrix = _trial_matrix(trial, self._channels)
            rows.append([matrix[index, offset] for index in self._kept for offset in range(3)])
        return np.asarray(rows, dtype=np.float64)


class CspEncoder(FeatureEncoder):
    def __init__(self, pairs: int = 2) -> None:
        self._pairs = pairs

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        left = [trial for trial, label in zip(trials, labels) if label == "left"]
        right = [trial for trial, label in zip(trials, labels) if label == "right"]
        self._filters = spatial_filters(left, right, pairs=self._pairs)

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([log_variances(trial, self._filters) for trial in trials], axis=0)


class FbcspEncoder(FeatureEncoder):
    def __init__(
        self,
        sample_rate_hz: float,
        *,
        pairs: int = 2,
        low_hz: float = 4.0,
        high_hz: float = 40.0,
        band_hz: float = 4.0,
        order: int = 4,
        keep: int = 4,
    ) -> None:
        self._rate = float(sample_rate_hz)
        self._pairs = pairs
        self._low = low_hz
        self._high = high_hz
        self._width = band_hz
        self._order = order
        self._keep = keep
        self._filters: list[tuple[float, float, np.ndarray]] = []
        self._chosen: list[int] = []

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        columns: list[np.ndarray] = []
        self._filters = []
        for low, high in _bands_for(self._rate, self._low, self._high, self._width):
            band = [_bandpass(trial, self._rate, low, high, self._order) for trial in trials]
            left = [trial for trial, label in zip(band, labels) if label == "left"]
            right = [trial for trial, label in zip(band, labels) if label == "right"]
            filters = spatial_filters(left, right, pairs=self._pairs)
            self._filters.append((low, high, filters))
            block = np.stack([log_variances(trial, filters) for trial in band], axis=0)
            for index in range(block.shape[1]):
                columns.append(block[:, index])
        if not columns:
            raise ValueError("FBCSP found no band under this sample rate")
        chosen = _most_informative(np.column_stack(columns), labels, self._keep)
        self._chosen = chosen

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        columns: list[np.ndarray] = []
        for low, high, filters in self._filters:
            band = [_bandpass(trial, self._rate, low, high, self._order) for trial in trials]
            block = np.stack([log_variances(trial, filters) for trial in band], axis=0)
            for index in range(block.shape[1]):
                columns.append(block[:, index])
        values = np.column_stack(columns)
        return values[:, self._chosen]


def make_encoder(
    method: str,
    sample_rate_hz: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
) -> FeatureEncoder:
    """Build the encoder for one feature method."""
    name = method.strip().lower()
    tuned = params or ModelParams()
    if name == "ts":
        return TimeSeriesEncoder(tuned.ts_points)
    if name == "psd":
        return PowerEncoder(sample_rate_hz, channels, tuned.psd_bands())
    if name == "tdp":
        return TimeDomainEncoder(channels)
    if name == "csp":
        return CspEncoder(tuned.csp_pairs)
    if name == "fbcsp":
        return FbcspEncoder(
            sample_rate_hz,
            pairs=tuned.csp_pairs,
            low_hz=tuned.fbcsp_low_hz,
            high_hz=tuned.fbcsp_high_hz,
            band_hz=tuned.fbcsp_band_hz,
            order=tuned.fbcsp_order,
            keep=tuned.fbcsp_keep,
        )
    raise ValueError(f"Unknown feature method {method!r}")
