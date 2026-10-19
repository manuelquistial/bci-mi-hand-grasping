"""The batch scores many classifiers without changing the pre-model request."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.batch import iter_jobs, run_batch, write_batch
from pipeline.classify import SubjectTrials, evaluate_classifier
from pipeline.service import PipelineService


def _trial(left_scale: float, right_scale: float, rng: np.random.Generator) -> np.ndarray:
    times = np.arange(160) / 160.0
    tone = np.sin(2 * np.pi * 10 * times)
    noise = 0.05 * rng.normal(size=(2, times.size))
    return np.vstack([left_scale * tone, right_scale * tone]) + noise


def _subject(name: str, rng: np.random.Generator) -> SubjectTrials:
    trials = []
    labels = []
    for _ in range(6):
        trials.append(_trial(3.0, 0.2, rng))
        labels.append("left")
        trials.append(_trial(0.2, 3.0, rng))
        labels.append("right")
    return SubjectTrials(name, trials, labels)


def _service(tmp_path: Path) -> PipelineService:
    rate = 160.0
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
            noise = 0.05 * rng.normal(size=160)
            left = np.column_stack([scale * tone[:160] + noise, 0.2 * tone[:160]])
            right = np.column_stack([0.2 * tone[:160], scale * tone[:160] + noise])
            rows.append({"kind": "trial_start", "label": "left", "samples": left.tolist()})
            rows.append({"kind": "trial_start", "label": "right", "samples": right.tolist()})
        return {"sample_rate_hz": rate, "channels": ["C3", "C4"], "trials": rows}

    return PipelineService(
        tmp_path,
        window_loader=lambda **_kwargs: window,
        trial_loader=trials,
        subject_lister=lambda _dataset: [
            {"id": "S001", "label": "S001"},
            {"id": "S002", "label": "S002"},
        ],
        subject_chooser=lambda items: items[0],
    )


def test_parallel_folds_match_the_single_thread() -> None:
    groups = [_subject(name, np.random.default_rng(2)) for name in ("S001", "S002", "S003")]
    single = evaluate_classifier(
        "lda",
        "loso",
        "csp",
        groups,
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
        workers=1,
    )
    parallel = evaluate_classifier(
        "lda",
        "loso",
        "csp",
        groups,
        sample_rate_hz=160.0,
        channels=["C3", "C4"],
        workers=3,
    )
    assert parallel.matrix == single.matrix
    assert parallel.accuracy == single.accuracy
    assert [item.name for item in parallel.folds] == ["S001", "S002", "S003"]


def test_riemann_is_not_repeated_for_every_feature() -> None:
    jobs = iter_jobs(
        ["lda", "riemann"],
        ["csp", "fbcsp"],
        ["cross_validation", "loso"],
    )
    assert jobs == [
        ("csp", "cross_validation", "lda"),
        ("csp", "loso", "lda"),
        ("fbcsp", "cross_validation", "lda"),
        ("fbcsp", "loso", "lda"),
        ("none", "cross_validation", "riemann"),
        ("none", "loso", "riemann"),
    ]


def test_batch_shares_one_cohort_and_leaves_the_preview_alone(tmp_path: Path) -> None:
    service = _service(tmp_path)
    results = run_batch(
        service,
        datasets=("shuqfa2024",),
        classifiers=("lda", "riemann"),
        features=("csp",),
        evaluations=("cross_validation",),
        workers=2,
    )
    assert [(row["features"], row["classifier"]) for row in results] == [
        ("csp", "lda"),
        ("none", "riemann"),
    ]
    assert all(row["error"] == "" for row in results)
    assert all(row["participants"] == 2 for row in results)
    assert results[0]["accuracy"] > 0.5
    preview = service.preview_filter(
        dataset="shuqfa2024",
        subject="S999",
        session_dir=None,
        notch_hz=50,
        band_low_hz=8,
        band_high_hz=30,
        order=4,
        features="csp",
        classifier="lda",
        evaluation="cross_validation",
    )
    assert preview["classifier"] == "lda"
    assert preview["subject"] == "S001"
    assert preview["participants"] == 2
    assert "2 participants" in preview["classifier_detail"]
    destination = tmp_path / "batch.json"
    write_batch(destination, results)
    assert "lda" in destination.read_text(encoding="utf-8")
