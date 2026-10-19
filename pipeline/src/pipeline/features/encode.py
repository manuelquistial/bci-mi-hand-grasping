"""Fit a descriptor on training trials and apply it to any trial from that fold."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline.features import cache
from pipeline.features.csp import (
    assert_hand_pair,
    filters_from_covariances,
    log_variances,
    log_variances_from_covariance,
    spatial_filters,
)
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
        frames = []
        for trial in trials:
            cached = cache.time_series(trial, self._points)
            series = downsample(trial, self._points) if cached is None else cached
            frames.append(np.asarray(series).ravel())
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
        rows = [
            cache.psd_row(trial, self._rate, self._channels, self._bands) for trial in trials
        ]
        if all(row is not None for row in rows):
            return np.stack(rows, axis=0)
        _names, values = band_powers(trials, self._rate, self._channels, self._bands)
        return values


class TimeDomainEncoder(FeatureEncoder):
    def __init__(self, channels: Sequence[str]) -> None:
        self._channels = list(channels)
        self._kept: list[int] = []

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        matrices = [self._matrix(trial) for trial in trials]
        self._kept = _keep_channels(matrices, labels)
        self.kept = [
            self._channels[index] for index in self._kept if index < len(self._channels)
        ]

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for trial in trials:
            matrix = self._matrix(trial)
            rows.append([matrix[index, offset] for index in self._kept for offset in range(3)])
        return np.asarray(rows, dtype=np.float64)

    def _matrix(self, trial: np.ndarray) -> np.ndarray:
        cached = cache.tdp_matrix(trial, self._channels)
        if cached is not None:
            return cached
        return _trial_matrix(trial, self._channels)


class CspEncoder(FeatureEncoder):
    def __init__(self, pairs: int = 2) -> None:
        self._pairs = pairs

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        assert_hand_pair(labels)
        stored = [cache.covariance_of(trial) for trial in trials]
        if all(item is not None for item in stored):
            left = [item[0] for item, label in zip(stored, labels) if label == "left"]
            right = [item[0] for item, label in zip(stored, labels) if label == "right"]
            if not left or not right:
                raise ValueError("CSP needs left and right trials")
            self._filters = filters_from_covariances(
                sum(left) / len(left),
                sum(right) / len(right),
                pairs=self._pairs,
            )
            return
        left = [trial for trial, label in zip(trials, labels) if label == "left"]
        right = [trial for trial, label in zip(trials, labels) if label == "right"]
        self._filters = spatial_filters(left, right, pairs=self._pairs)

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for trial in trials:
            stored = cache.covariance_of(trial)
            if stored is None:
                rows.append(log_variances(trial, self._filters))
            else:
                covariance, count = stored
                rows.append(log_variances_from_covariance(covariance, count, self._filters))
        return np.stack(rows, axis=0)


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
        assert_hand_pair(labels)
        stored = [
            cache.fbcsp_bands(trial, self._rate, self._low, self._high, self._width, self._order)
            for trial in trials
        ]
        if all(item is not None for item in stored):
            self._fit_cached(stored, labels)
            return
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
        stored = [
            cache.fbcsp_bands(trial, self._rate, self._low, self._high, self._width, self._order)
            for trial in trials
        ]
        if all(item is not None for item in stored):
            columns = self._columns_from_cache(stored)
        else:
            columns = []
            for low, high, filters in self._filters:
                band = [_bandpass(trial, self._rate, low, high, self._order) for trial in trials]
                block = np.stack([log_variances(trial, filters) for trial in band], axis=0)
                for index in range(block.shape[1]):
                    columns.append(block[:, index])
        values = np.column_stack(columns)
        return values[:, self._chosen]

    def _fit_cached(
        self,
        stored: Sequence[list[tuple[float, float, np.ndarray, int]]],
        labels: Sequence[str],
    ) -> None:
        columns: list[np.ndarray] = []
        self._filters = []
        if not stored or not stored[0]:
            raise ValueError("FBCSP found no band under this sample rate")
        for band_index, (low, high, _cov, _count) in enumerate(stored[0]):
            left = [item[band_index][2] for item, label in zip(stored, labels) if label == "left"]
            right = [item[band_index][2] for item, label in zip(stored, labels) if label == "right"]
            if not left or not right:
                raise ValueError("FBCSP needs left and right trials")
            filters = filters_from_covariances(
                sum(left) / len(left),
                sum(right) / len(right),
                pairs=self._pairs,
            )
            self._filters.append((low, high, filters))
            block = np.stack(
                [
                    log_variances_from_covariance(item[band_index][2], item[band_index][3], filters)
                    for item in stored
                ],
                axis=0,
            )
            for index in range(block.shape[1]):
                columns.append(block[:, index])
        chosen = _most_informative(np.column_stack(columns), labels, self._keep)
        self._chosen = chosen

    def _columns_from_cache(
        self,
        stored: Sequence[list[tuple[float, float, np.ndarray, int]]],
    ) -> list[np.ndarray]:
        columns: list[np.ndarray] = []
        for band_index, (_low, _high, filters) in enumerate(self._filters):
            block = np.stack(
                [
                    log_variances_from_covariance(item[band_index][2], item[band_index][3], filters)
                    for item in stored
                ],
                axis=0,
            )
            for index in range(block.shape[1]):
                columns.append(block[:, index])
        return columns


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
