"""Classifiers and the three evaluation choices."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.classify import SubjectTrials, evaluate_classifier
from pipeline.service import PipelineService


def _trial(left_scale: float, right_scale: float, rng: np.random.Generator) -> np.ndarray:
    times = np.arange(80) / 160.0
    tone = np.sin(2 * np.pi * 10 * times)
    noise = 0.05 * rng.normal(size=(2, times.size))
    return np.vstack([left_scale * tone, right_scale * tone]) + noise


def _subject(name: str, rng: np.random.Generator, count: int = 8) -> SubjectTrials:
    trials = []
    labels = []
    for _ in range(count):
        trials.append(_trial(3.0, 0.2, rng))
        labels.append("left")
        trials.append(_trial(0.2, 3.0, rng))
        labels.append("right")
    return SubjectTrials(name, trials, labels)


def test_cross_validation_separates_the_two_classes() -> None:
    group = _subject("S001", np.random.default_rng(0))
    result = evaluate_classifier(
        "lda",
        "cross_validation",
        "csp",
        [group],
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
    )
    assert result.accuracy > 0.8
    assert result.metrics["recall"] == result.metrics["sensitivity"]
    assert result.metrics["kappa"] is not None and result.metrics["kappa"] > 0.5
    assert result.metrics["f1"] is not None and result.metrics["f1"] > 0.8
    assert result.metrics["auc"] is not None
    assert len(result.matrix) == 2
    assert len(result.folds) == 5
    assert result.folds[0].name == "fold 1"


def test_split_reports_train_validation_and_test() -> None:
    group = _subject("S001", np.random.default_rng(1), count=12)
    result = evaluate_classifier(
        "logistic_regression",
        "split",
        "psd",
        [group],
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
    )
    assert [item.name for item in result.folds] == ["train", "validation", "test"]
    assert result.accuracy == result.folds[2].accuracy
    assert result.accuracy > 0.7


def test_loso_holds_out_each_participant() -> None:
    rng = np.random.default_rng(2)
    groups = [_subject(name, rng, count=6) for name in ("S001", "S002", "S003")]
    result = evaluate_classifier(
        "riemann",
        "loso",
        "none",
        groups,
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
    )
    assert result.accuracy > 0.7
    assert [item.name for item in result.folds] == ["S001", "S002", "S003"]


def test_vector_classifier_needs_a_feature_method() -> None:
    group = _subject("S001", np.random.default_rng(3), count=4)
    with pytest.raises(ValueError, match="feature method"):
        evaluate_classifier(
            "lda",
            "cross_validation",
            "none",
            [group],
            sample_rate_hz=160.0,
            channels=["C3", "C4"],
        )


def test_ensembles_and_margin_models_fit() -> None:
    group = _subject("S001", np.random.default_rng(4), count=8)
    for name in (
        "slda",
        "linear_svm",
        "knn",
        "gaussian_nb",
        "decision_tree",
        "svm_rbf",
        "random_forest",
        "random_subspace_knn",
        "boosted_trees",
        "bagged_trees",
        "rusboosted_trees",
    ):
        result = evaluate_classifier(
            name,
            "cross_validation",
            "ts",
            [group],
            sample_rate_hz=160.0,
            channels=["C3", "C4"],
        )
        assert 0.0 <= result.accuracy <= 1.0


def test_preview_reports_the_chosen_evaluation(tmp_path) -> None:
    from pathlib import Path

    rate = 250.0
    times = np.arange(0, 1.0, 1.0 / rate)
    tone = np.sin(2 * np.pi * 12 * times)
    window = {
        "sample_rate_hz": rate,
        "channels": ["C3", "C4"],
        "samples": np.column_stack([tone, 0.2 * tone]).tolist(),
    }

    def trials(**kwargs: object) -> dict:
        subject = str(kwargs.get("subject") or "S001")
        scale = 3.0 if subject == "S001" else 2.5
        rng = np.random.default_rng(0 if subject == "S001" else 1)
        rows = []
        for _ in range(6):
            noise = 0.05 * rng.normal(size=80)
            left = np.column_stack([scale * tone[:80] + noise, 0.2 * tone[:80]])
            right = np.column_stack([0.2 * tone[:80], scale * tone[:80] + noise])
            rows.append({"kind": "trial_start", "label": "left", "samples": left.tolist()})
            rows.append({"kind": "trial_start", "label": "right", "samples": right.tolist()})
        return {"sample_rate_hz": rate, "channels": ["C3", "C4"], "trials": rows}

    service = PipelineService(
        Path(tmp_path),
        window_loader=lambda **_kwargs: window,
        trial_loader=trials,
        subject_lister=lambda _dataset: [
            {"id": "S001", "label": "S001"},
            {"id": "S002", "label": "S002"},
        ],
        subject_chooser=lambda items: items[0],
    )
    common = dict(
        dataset="shuqfa2024",
        subject="S999",
        session_dir=None,
        notch_hz=50,
        band_low_hz=8,
        band_high_hz=30,
        order=4,
        features="csp",
        classifier="lda",
    )
    crossed = service.preview_filter(**common, evaluation="cross_validation")
    assert crossed["classifier"] == "lda"
    assert crossed["subject"] == "S001"
    assert crossed["participants"] == 2
    assert "2 participants" in crossed["classifier_detail"]
    assert "2 participants" in crossed["feature_detail"]
    assert crossed["evaluation"] == "cross_validation"
    assert crossed["accuracy"] > 0.5
    assert len(crossed["classifier_folds"]) >= 2
    assert set(crossed["metrics"]) == {
        "accuracy",
        "kappa",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
        "mcc",
    }
    assert len(crossed["confusion_matrix"]) == 2
    held = service.preview_filter(**common, evaluation="loso")
    assert [item["name"] for item in held["classifier_folds"]] == ["S001", "S002"]
    assert held["accuracy"] > 0.5


def test_hyperparameters_reach_the_estimator() -> None:
    from pipeline.classify.models import build_model
    from pipeline.params import model_params

    parsed = model_params({"regularization_c": 0.25, "gamma": 0.1, "folds": 3})
    assert parsed.regularization_c == 0.25
    assert parsed.folds == 3
    assert parsed.gamma == 0.1
    model = build_model("svm_rbf", 8, parsed)
    assert model.C == 0.25
    assert model.gamma == 0.1


def test_multiclass_metrics_follow_the_confusion_matrix() -> None:
    from pipeline.classify.metrics import summarize

    true = np.array(["left", "left", "right", "right", "feet", "feet"])
    predicted = np.array(["left", "right", "right", "right", "feet", "left"])
    labels, matrix, metrics = summarize(true, predicted, None)
    assert labels == ["feet", "left", "right"]
    assert matrix[labels.index("left")][labels.index("right")] == 1
    assert metrics["sensitivity"] == metrics["recall"]
    assert metrics["auc"] is None
    assert metrics["mcc"] is not None
    assert 0 <= metrics["specificity"] <= 1
