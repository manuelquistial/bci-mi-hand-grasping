"""Prepared trial descriptors match the descriptors computed from the samples."""

from __future__ import annotations

import numpy as np

from pipeline.classify import SubjectTrials
from pipeline.classify.riemann import TangentEncoder
from pipeline.features.cache import clear, prepare
from pipeline.features.csp import log_variances, log_variances_from_covariance
from pipeline.features.encode import CspEncoder, FbcspEncoder, PowerEncoder
from pipeline.alignment import trial_covariance


def _trials() -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(0)
    trials = []
    labels = []
    for _ in range(4):
        trials.append(rng.normal(size=(4, 80)) + np.array([[0.4], [0.0], [0.0], [0.0]]))
        labels.append("left")
        trials.append(rng.normal(size=(4, 80)) + np.array([[0.0], [0.4], [0.0], [0.0]]))
        labels.append("right")
    return trials, labels


def test_covariance_log_variance_matches_the_projected_samples() -> None:
    rng = np.random.default_rng(1)
    trial = rng.normal(size=(5, 40)) + 2.0
    filters = rng.normal(size=(5, 2))
    direct = log_variances(trial, filters)
    cached = log_variances_from_covariance(trial_covariance(trial), trial.shape[1], filters)
    assert np.allclose(direct, cached)


def test_prepared_descriptors_match_a_fresh_fit() -> None:
    trials, labels = _trials()
    channels = ["C3", "Cz", "C4", "Pz"]
    raw_csp = CspEncoder(2)
    raw_csp.fit(trials, labels)
    raw_power = PowerEncoder(160.0, channels)
    raw_fbcsp = FbcspEncoder(160.0, pairs=1, keep=4)
    raw_fbcsp.fit(trials, labels)
    raw_tangent = TangentEncoder()
    raw_tangent.fit(trials, labels)
    expected = {
        "csp": raw_csp.transform(trials),
        "psd": raw_power.transform(trials),
        "fbcsp": raw_fbcsp.transform(trials),
        "riemann": raw_tangent.transform(trials),
    }
    prepare([SubjectTrials("S001", trials, labels)], 160.0, channels)
    try:
        csp = CspEncoder(2)
        csp.fit(trials, labels)
        power = PowerEncoder(160.0, channels)
        fbcsp = FbcspEncoder(160.0, pairs=1, keep=4)
        fbcsp.fit(trials, labels)
        tangent = TangentEncoder()
        tangent.fit(trials, labels)
        assert np.allclose(csp.transform(trials), expected["csp"])
        assert np.allclose(power.transform(trials), expected["psd"])
        assert np.allclose(fbcsp.transform(trials), expected["fbcsp"])
        assert np.allclose(tangent.transform(trials), expected["riemann"])
    finally:
        clear()
