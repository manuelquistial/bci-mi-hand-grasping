"""Tests for the multi-channel ring buffer and channel parsing."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.buffer import RingBuffer
from pipeline.visualizer import _parse_channels


def test_ring_buffer_append_and_order() -> None:
    buf = RingBuffer(n_channels=2, capacity=4)
    buf.append([[1.0, 10.0], [2.0, 20.0]])
    assert buf.size == 2
    latest = buf.latest()
    assert latest.shape == (2, 2)
    np.testing.assert_allclose(latest[0], [1.0, 2.0])
    np.testing.assert_allclose(latest[1], [10.0, 20.0])


def test_ring_buffer_wraps() -> None:
    buf = RingBuffer(n_channels=1, capacity=3)
    buf.append([[1.0], [2.0], [3.0], [4.0], [5.0]])
    latest = buf.latest()
    assert buf.size == 3
    np.testing.assert_allclose(latest[0], [3.0, 4.0, 5.0])


def test_ring_buffer_rejects_bad_shape() -> None:
    buf = RingBuffer(n_channels=2, capacity=8)
    with pytest.raises(ValueError, match="expected"):
        buf.append([1.0, 2.0, 3.0])


def test_parse_channels_ranges() -> None:
    assert _parse_channels(None, 4) == [0, 1, 2, 3]
    assert _parse_channels("0,2", 4) == [0, 2]
    assert _parse_channels("1-3", 4) == [1, 2, 3]
    assert _parse_channels("0,1-2,1", 4) == [0, 1, 2]


def test_parse_channels_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _parse_channels("0,5", 4)
