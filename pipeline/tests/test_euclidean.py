"""Euclidean alignment sends the mean trial covariance to the identity."""

from __future__ import annotations

import numpy as np

from pipeline.alignment import align_trials, apply_alignment, mean_covariance


def test_aligned_trials_average_to_the_identity() -> None:
    rng = np.random.default_rng(0)
    left = np.array([[3.0, 0.4], [0.2, 0.5]]) @ rng.normal(size=(2, 400))
    right = np.array([[0.6, -0.3], [0.1, 2.5]]) @ rng.normal(size=(2, 400))
    aligned, weights = align_trials([left, right])
    assert weights.shape == (2, 2)
    assert np.allclose(mean_covariance(aligned), np.eye(2), atol=1e-6)


def test_apply_alignment_matches_the_trial_transform() -> None:
    trial = np.arange(12, dtype=np.float64).reshape(3, 4)
    aligned, weights = align_trials([trial, trial + 1])
    restored = apply_alignment(trial.T, weights)
    assert np.allclose(restored, aligned[0].T)
