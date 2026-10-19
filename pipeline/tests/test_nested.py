"""Nested cross-subject splits stay on the participant, not the trial."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.classify import SubjectTrials, evaluate_classifier
from pipeline.classify.nested import evaluate_nested
from pipeline.classify.splits import build_manifest, check_manifest
from pipeline.features.csp import csp_features
from pipeline.params import ModelParams


def _trial(left: float, right: float, rng: np.random.Generator) -> np.ndarray:
    times = np.arange(80) / 160.0
    tone = np.sin(2 * np.pi * 10 * times)
    noise = 0.05 * rng.normal(size=(2, times.size))
    return np.vstack([left * tone, right * tone]) + noise


def _people(dataset: str) -> list[SubjectTrials]:
    groups = []
    for index, name in enumerate(("S001", "S002", "S003", "S004")):
        rng = np.random.default_rng(index + 1)
        trials = []
        labels = []
        for _ in range(4):
            trials.append(_trial(3.0, 0.2, rng))
            labels.append("left")
            trials.append(_trial(0.2, 3.0, rng))
            labels.append("right")
        groups.append(
            SubjectTrials(
                name,
                trials,
                labels,
                dataset,
                "02",
                tuple(f"{dataset}:{name}:02:{trial}" for trial in range(len(labels))),
            )
        )
    return groups


def test_each_participant_is_an_outer_test_once() -> None:
    manifest = build_manifest(
        _people("shuqfa2024"),
        dataset_id="shuqfa2024",
        outer="group",
        outer_folds=5,
        inner_folds=3,
        seed=42,
    )
    check_manifest(manifest)
    assert manifest["outer_folds"] == 4
    assert "reduced from 5 to 4" in manifest["adjustment"]
    seen = [name for fold in manifest["folds"] for name in fold["test_subjects"]]
    assert sorted(seen) == ["S001", "S002", "S003", "S004"]
    for fold in manifest["folds"]:
        test = set(fold["test_subjects"])
        for inner in fold["inner"]:
            assert test.isdisjoint(inner["train_subjects"])
            assert test.isdisjoint(inner["validation_subjects"])


def test_nested_reuses_the_manifest_for_every_pipeline() -> None:
    groups = _people("liu2024_stroke")
    params = ModelParams(folds=4, inner_folds=2, search_budget=1)
    manifest = build_manifest(
        groups,
        dataset_id="liu2024_stroke",
        outer="group",
        outer_folds=4,
        inner_folds=2,
        seed=7,
    )
    first = evaluate_nested(
        "lda",
        "ts",
        groups,
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
        params=params,
        manifest=manifest,
    )
    second = evaluate_classifier(
        "lda",
        "nested",
        "csp",
        groups,
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
        params=params,
        manifest=manifest,
    )
    assert first.report is not None and second.report is not None
    assert first.report["manifest"]["folds"] == second.report["manifest"]["folds"]
    assert first.metrics["balanced_accuracy"] is not None
    subjects = {row["subject_id"] for row in first.report["subjects"]}
    assert subjects == {"S001", "S002", "S003", "S004"}
    datasets = {row["dataset_id"] for row in first.report["predictions"]}
    assert datasets == {"liu2024_stroke"}
    assert first.report["search_seconds"] >= 0
    assert first.report["refit_seconds"] >= 0


def test_csp_does_not_drop_extra_classes() -> None:
    rng = np.random.default_rng(0)
    trials = [_trial(3.0, 0.2, rng), _trial(0.2, 3.0, rng), _trial(1.0, 1.0, rng)]
    with pytest.raises(ValueError, match="Other classes"):
        csp_features(trials, ["left", "right", "both_feet"])
