"""Fixed-size ring buffer for multi-channel LSL samples."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Circular buffer with shape ``(n_channels, capacity)``."""

    def __init__(self, n_channels: int, capacity: int) -> None:
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.n_channels = n_channels
        self.capacity = capacity
        self._data = np.zeros((n_channels, capacity), dtype=np.float64)
        self._write = 0
        self._filled = 0

    @property
    def size(self) -> int:
        return self._filled

    def append(self, samples: np.ndarray) -> None:
        """
        Append samples with shape ``(n_samples, n_channels)`` or
        ``(n_channels,)`` for a single sample.
        """
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != self.n_channels:
                raise ValueError(
                    f"expected {self.n_channels} channels, got {arr.shape[0]}"
                )
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] != self.n_channels:
                raise ValueError(
                    f"expected shape (*, {self.n_channels}), got {arr.shape}"
                )
        else:
            raise ValueError(f"samples must be 1D or 2D, got ndim={arr.ndim}")

        n = arr.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            self._data[:] = arr[-self.capacity :].T
            self._write = 0
            self._filled = self.capacity
            return

        end = self._write + n
        if end <= self.capacity:
            self._data[:, self._write : end] = arr.T
        else:
            first = self.capacity - self._write
            self._data[:, self._write :] = arr[:first].T
            self._data[:, : n - first] = arr[first:].T
        self._write = (self._write + n) % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def latest(self) -> np.ndarray:
        """Return samples in chronological order, shape ``(n_channels, size)``."""
        if self._filled == 0:
            return np.zeros((self.n_channels, 0), dtype=np.float64)
        if self._filled < self.capacity:
            return self._data[:, : self._filled].copy()
        return np.concatenate(
            (self._data[:, self._write :], self._data[:, : self._write]),
            axis=1,
        )
