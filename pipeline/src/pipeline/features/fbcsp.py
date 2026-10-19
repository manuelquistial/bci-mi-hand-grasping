"""Filter-bank CSP, separate from the single-band CSP flow.

The aligned trial is split into contiguous 4 Hz bands from 4 to 40 Hz with a
Butterworth filter. Each band has its own CSP. Mutual information then keeps
the four features that tell left from right most clearly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, filtfilt

from pipeline.features.csp import assert_hand_pair, log_variances, spatial_filters

_ORDER = 4
_KEEP = 4


def fbcsp_features(
    trials: Sequence[np.ndarray],
    labels: Sequence[str],
    sample_rate_hz: float,
    *,
    pairs: int = 2,
    low_hz: float = 4.0,
    high_hz: float = 40.0,
    band_hz: float = 4.0,
    order: int = _ORDER,
    keep: int = _KEEP,
) -> tuple[list[str], np.ndarray]:
    """Return ``(feature names, values)`` after mutual-information selection."""
    if len(labels) != len(trials) or not trials:
        raise ValueError("FBCSP needs one label per trial")
    assert_hand_pair(labels)
    rate = float(sample_rate_hz)
    names: list[str] = []
    columns: list[np.ndarray] = []
    for low, high in _bands_for(rate, low_hz, high_hz, band_hz):
        band = [_bandpass(trial, rate, low, high, order) for trial in trials]
        left = [trial for trial, label in zip(band, labels) if label == "left"]
        right = [trial for trial, label in zip(band, labels) if label == "right"]
        if not left or not right:
            raise ValueError("FBCSP needs left and right trials")
        filters = spatial_filters(left, right, pairs=pairs)
        block = np.stack([log_variances(trial, filters) for trial in band], axis=0)
        for index in range(block.shape[1]):
            names.append(f"{low:.0f}-{high:.0f} Hz {index + 1}")
            columns.append(block[:, index])
    if not columns:
        raise ValueError("FBCSP found no band under this sample rate")
    values = np.column_stack(columns)
    chosen = _most_informative(values, labels, keep)
    return [names[index] for index in chosen], values[:, chosen]


def _bands_for(
    rate: float,
    low_hz: float = 4.0,
    high_hz: float = 40.0,
    band_hz: float = 4.0,
) -> list[tuple[float, float]]:
    nyquist = rate / 2.0
    width = float(band_hz)
    if width <= 0:
        raise ValueError("FBCSP band width must be positive")
    start = float(low_hz)
    bands: list[tuple[float, float]] = []
    while start + width <= float(high_hz) + 1e-9:
        bands.append((start, start + width))
        start += width
    return [(low, high) for low, high in bands if high < nyquist]


def _bandpass(
    trial: np.ndarray,
    rate: float,
    low: float,
    high: float,
    order: int = _ORDER,
) -> np.ndarray:
    array = np.asarray(trial, dtype=np.float64)
    numerator, denominator = butter(order, [low, high], btype="bandpass", fs=rate)
    filtered = filtfilt(numerator, denominator, array, axis=1)
    return np.asarray(filtered, dtype=np.float64)


def _most_informative(values: np.ndarray, labels: Sequence[str], keep: int) -> list[int]:
    scores = [_mutual_information(values[:, index], labels) for index in range(values.shape[1])]
    order = np.argsort(np.asarray(scores), kind="mergesort")[::-1]
    count = min(keep, len(order))
    return [int(index) for index in order[:count]]


def _mutual_information(feature: np.ndarray, labels: Sequence[str]) -> float:
    series = np.asarray(feature, dtype=np.float64)
    if np.allclose(series, series[0]):
        return 0.0
    bins = min(8, max(2, len(series) // 2))
    edges = np.linspace(float(series.min()), float(series.max()), bins + 1)
    codes = np.clip(np.digitize(series, edges[1:-1]), 0, bins - 1)
    classes = list(dict.fromkeys(labels))
    total = len(series)
    information = 0.0
    for code in range(bins):
        in_bin = codes == code
        bin_count = int(np.count_nonzero(in_bin))
        if bin_count == 0:
            continue
        for label in classes:
            joint = int(np.count_nonzero(in_bin & (np.asarray(labels) == label)))
            if joint == 0:
                continue
            label_count = sum(item == label for item in labels)
            information += (joint / total) * np.log(
                (joint * total) / (bin_count * label_count)
            )
    return float(information)
