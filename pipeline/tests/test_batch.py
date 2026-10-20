"""The batch scores many classifiers without changing the pre-model request."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import json

from pipeline.batch import comparison_rows, iter_jobs, run_batch, write_batch
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


def _bump(value: int) -> int:
    return value + 1


def test_fork_pool_can_run_without_recycling_workers() -> None:
    import inspect

    from pipeline.batch import _linux_executor

    assert "max_tasks_per_child" not in inspect.getsource(_linux_executor)
    with _linux_executor(1) as pool:
        assert pool.submit(_bump, 1).result(timeout=30) == 2


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
        ("covariance", "cross_validation", "riemann"),
        ("covariance", "loso", "riemann"),
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
        ("covariance", "riemann"),
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
    assert "confusion_matrix" in destination.read_text(encoding="utf-8")


def test_saved_run_keeps_matrices_timing_and_the_best_row_first(tmp_path: Path) -> None:
    service = _service(tmp_path)
    destination = tmp_path / "batch.json"
    loads = {"count": 0}
    original = service.load_cohort

    def counting(**kwargs: object) -> dict:
        loads["count"] += 1
        return original(**kwargs)

    service.load_cohort = counting  # type: ignore[method-assign]
    results = run_batch(
        service,
        datasets=("shuqfa2024",),
        classifiers=("lda", "gaussian_nb"),
        features=("csp",),
        evaluations=("cross_validation",),
        workers=2,
        output=destination,
    )
    assert loads["count"] == 1
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["elapsed_seconds"] >= 0
    assert saved["datasets"][0]["load_seconds"] >= 0
    covariance = saved["datasets"][0]["covariances"]
    assert [item["subject"] for item in covariance] == ["S001", "S002"]
    assert len(covariance[0]["matrix"]) == 2
    assert len(covariance[0]["matrix"][0]) == 2
    assert results[0]["confusion_matrix"]
    assert results[0]["seconds"] >= 0
    table = destination.with_name("batch-comparison.csv").read_text(encoding="utf-8")
    assert table.splitlines()[1].startswith("1,")
    run_batch(
        service,
        datasets=("shuqfa2024",),
        classifiers=("lda", "gaussian_nb"),
        features=("csp",),
        evaluations=("cross_validation",),
        workers=2,
        output=destination,
    )
    assert loads["count"] == 1


def test_comparison_puts_the_higher_accuracy_first() -> None:
    rows = comparison_rows(
        [
            {
                "dataset": "shuqfa2024",
                "features": "csp",
                "evaluation": "loso",
                "classifier": "lda",
                "accuracy": 0.4,
                "participants": 2,
                "seconds": 3.5,
                "metrics": {"accuracy": 0.4, "kappa": 0.1, "f1": 0.4},
                "confusion_labels": ["left", "right"],
                "confusion_matrix": [[1, 1], [1, 1]],
                "error": "",
            },
            {
                "dataset": "shuqfa2024",
                "features": "csp",
                "evaluation": "loso",
                "classifier": "svm_rbf",
                "accuracy": 0.9,
                "participants": 2,
                "seconds": 8.0,
                "metrics": {"accuracy": 0.9, "kappa": 0.8, "f1": 0.9},
                "confusion_labels": ["left", "right"],
                "confusion_matrix": [[4, 0], [1, 3]],
                "error": "",
            },
        ]
    )
    assert rows[0]["classifier"] == "svm_rbf"
    assert rows[0]["rank"] == 1
    assert rows[0]["seconds"] == 8.0
    assert rows[0]["confusion_matrix"] == [[4, 0], [1, 3]]
    assert rows[1]["rank"] == 2
    assert rows[0]["spatial"] == "none"
    assert rows[0]["alignment"] == "none"


def test_batch_prepares_each_spatial_method_from_one_load(tmp_path: Path) -> None:
    service = _service(tmp_path)
    loads = {"count": 0}
    original = service.load_cohort

    def counting(**kwargs: object) -> dict:
        loads["count"] += 1
        assert kwargs["spatial"] == "none"
        assert kwargs["alignment"] == "none"
        return original(**kwargs)

    service.load_cohort = counting  # type: ignore[method-assign]
    results = run_batch(
        service,
        datasets=("shuqfa2024",),
        classifiers=("lda",),
        features=("csp",),
        evaluations=("cross_validation",),
        spatial=("none", "car"),
        alignment="none",
        workers=1,
    )
    assert loads["count"] == 1
    assert [(row["spatial"], row["alignment"], row["classifier"]) for row in results] == [
        ("none", "none", "lda"),
        ("car", "none", "lda"),
    ]
    assert all(row["error"] == "" for row in results)


def test_failed_row_is_retried_and_a_success_is_kept(tmp_path: Path) -> None:
    service = _service(tmp_path)
    destination = tmp_path / "batch.json"
    results = run_batch(
        service,
        datasets=("shuqfa2024",),
        classifiers=("lda", "gaussian_nb"),
        features=("csp",),
        evaluations=("cross_validation",),
        workers=1,
        output=destination,
    )
    kept = next(row for row in results if row["classifier"] == "gaussian_nb")
    saved = json.loads(destination.read_text(encoding="utf-8"))
    for row in saved["results"]:
        if row["classifier"] == "lda":
            row["error"] = "killed"
            row["metrics"] = {}
            row["accuracy"] = None
    destination.write_text(json.dumps(saved), encoding="utf-8")
    again = run_batch(
        service,
        datasets=("shuqfa2024",),
        classifiers=("lda", "gaussian_nb"),
        features=("csp",),
        evaluations=("cross_validation",),
        workers=1,
        output=destination,
    )
    lda = next(row for row in again if row["classifier"] == "lda")
    naive = next(row for row in again if row["classifier"] == "gaussian_nb")
    assert lda["error"] == ""
    assert lda["accuracy"] is not None
    assert naive["seconds"] == kept["seconds"]
    assert naive["accuracy"] == kept["accuracy"]
