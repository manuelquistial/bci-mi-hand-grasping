"""Log band power from a Welch spectrum: theta, mu, and beta."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import welch

_BANDS = (("theta", 4.0, 8.0), ("mu", 8.0, 12.0), ("beta", 13.0, 30.0))
_EPS = 1e-12


def band_powers(
    trials: Sequence[np.ndarray],
    sample_rate_hz: float,
    channels: Sequence[str],
    bands: Sequence[tuple[str, float, float]] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return ``(feature names, values)`` with one row per trial."""
    if not trials:
        raise ValueError("Band power needs at least one trial")
    rate = float(sample_rate_hz)
    if rate <= 0:
        raise ValueError("Sample rate must be positive")
    used = tuple(bands) if bands is not None else _BANDS
    names = [f"{channel} {band}" for channel in channels for band, _low, _high in used]
    rows = [_one(trial, rate, len(channels), used) for trial in trials]
    return names, np.stack(rows, axis=0)


def _one(
    trial: np.ndarray,
    rate: float,
    width: int,
    bands: Sequence[tuple[str, float, float]] = _BANDS,
) -> np.ndarray:
    array = np.asarray(trial, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != width or array.shape[1] < 8:
        raise ValueError("A spectral trial needs eight samples on every channel")
    features: list[float] = []
    segment = min(256, int(array.shape[1]))
    for channel in range(width):
        freqs, power = welch(array[channel], fs=rate, nperseg=segment)
        for _name, low, high in bands:
            chosen = (freqs >= low) & (freqs < high)
            if not np.any(chosen):
                features.append(np.log(_EPS))
                continue
            area = float(np.trapezoid(power[chosen], freqs[chosen]))
            features.append(float(np.log(area + _EPS)))
    return np.asarray(features, dtype=np.float64)
