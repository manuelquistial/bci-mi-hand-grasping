"""Spatial operators used after the temporal filter."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.spatial import apply_spatial
from pipeline.spatial.car import matrix as car_matrix
from pipeline.spatial.hjorth import matrix as hjorth_matrix
from pipeline.spatial.large_laplacian import matrix as large_matrix
from pipeline.spatial.spherical_spline import matrix as spline_matrix

_MOTOR = ["FC3", "C1", "C3", "C5", "CP3", "FCz", "Cz", "CPz", "C2", "C4", "FC4", "CP4"]


def test_car_is_the_centering_matrix() -> None:
    operator, labels = car_matrix(["C3", "Cz", "C4"])
    expected = np.eye(3) - np.ones((3, 3)) / 3
    assert labels == ["C3", "Cz", "C4"]
    assert np.allclose(operator, expected)


def test_none_leaves_the_filtered_signal_unchanged() -> None:
    samples = np.arange(12, dtype=np.float64).reshape(4, 3)
    out, labels = apply_spatial(samples, ["C3", "Cz", "C4"], "none")
    assert labels == ["C3", "Cz", "C4"]
    assert np.allclose(out, samples)


def test_hjorth_subtracts_the_neighbor_mean_and_drops_the_edge() -> None:
    operator, labels = hjorth_matrix(_MOTOR)
    assert "Cz" in labels
    assert "FCz" not in labels
    row = operator[labels.index("C3")]
    neighbors = [index for index, name in enumerate(_MOTOR) if name in {"FC3", "C1", "C5", "CP3"}]
    assert row[_MOTOR.index("C3")] == 1
    assert np.allclose(row[neighbors], -0.25)
    assert np.isclose(row.sum(), 0)


def test_large_laplacian_weights_sum_to_one() -> None:
    operator, labels = large_matrix(_MOTOR)
    assert "Cz" in labels
    row = operator[labels.index("Cz")]
    assert np.isclose(row.sum(), 0)
    assert row[_MOTOR.index("Cz")] == 1


def test_spline_of_a_constant_is_flat() -> None:
    operator, labels = spline_matrix(_MOTOR)
    assert "Cz" in labels
    flat = operator @ np.ones(len(_MOTOR))
    assert np.allclose(flat, flat[0], atol=1e-6)


def test_unnamed_channels_cannot_build_a_laplacian() -> None:
    with pytest.raises(ValueError, match="10-10"):
        hjorth_matrix(["Ch1", "Ch2", "Ch3", "Ch4"])


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown spatial method"):
        apply_spatial(np.ones((4, 2)), ["C3", "C4"], "wavelet")
