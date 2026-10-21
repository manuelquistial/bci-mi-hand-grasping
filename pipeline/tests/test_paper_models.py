"""MDM and TWFB + DGFMDRM stay outside the batch classifier list."""

from __future__ import annotations

import numpy as np

from pipeline.classify.evaluate import SubjectTrials
from pipeline.classify.models import CLASSIFIERS
from pipeline.classify.paper_models import (
    DiscriminantGeodesic,
    MinimumDistanceMean,
    TwfbSearch,
    _cores,
    _distance,
    _map_tasks,
    _project,
    _spd,
    apply_signal,
    comparison_row,
    csp_variants,
    evaluate_cross_subject,
    evaluate_csp,
    evaluate_csp_grid,
    score_csp_grid,
)
from pipeline.spatial import METHODS as SPATIAL_METHODS


def _trial(left_scale: float, right_scale: float, rng: np.random.Generator) -> np.ndarray:
    times = np.arange(int(4.0 * 160)) / 160.0
    tone = np.sin(2 * np.pi * 10 * times)
    noise = 0.05 * rng.normal(size=(2, times.size))
    return np.vstack([left_scale * tone, right_scale * tone]) + noise


def _hands(rng: np.random.Generator, count: int = 6) -> tuple[list[np.ndarray], list[str]]:
    trials: list[np.ndarray] = []
    labels: list[str] = []
    for _ in range(count):
        trials.append(_trial(3.0, 0.2, rng))
        labels.append("left")
        trials.append(_trial(0.2, 3.0, rng))
        labels.append("right")
    return trials, labels


def _subject(name: str, rng: np.random.Generator) -> SubjectTrials:
    trials, labels = _hands(rng)
    return SubjectTrials(name, trials, labels, "shuqfa2024")


def test_paper_models_are_not_part_of_the_batch_list() -> None:
    assert "mdm" not in CLASSIFIERS
    assert "twfb_dgfmdr" not in CLASSIFIERS
    assert "riemann" not in CLASSIFIERS


def test_identical_covariances_have_no_riemannian_distance() -> None:
    matrix = _spd(np.array([[2.0, 0.2], [0.2, 1.0]]))
    assert _distance(matrix, matrix) == 0.0


def test_mdm_separates_the_two_hands() -> None:
    trials, labels = _hands(np.random.default_rng(1))
    model = MinimumDistanceMean().fit(trials, labels)
    predicted = model.predict(trials)
    assert sum(left == right for left, right in zip(labels, predicted, strict=True)) >= 10


def test_twfb_keeps_a_window_from_the_supplied_grid() -> None:
    trials, labels = _hands(np.random.default_rng(2))
    model = DiscriminantGeodesic().fit(
        trials,
        labels,
        sample_rate_hz=160.0,
        window=(0.0, 1.0),
        band=(8.0, 12.0),
    )
    predicted = model.predict(trials)
    assert len(predicted) == len(labels)
    assert set(predicted) <= {"left", "right"}


def test_dgfmdr_uses_the_reduced_tangent_space_when_there_are_enough_channels() -> None:
    rng = np.random.default_rng(5)
    times = np.arange(int(4.0 * 160)) / 160.0
    tone = np.sin(2 * np.pi * 10 * times)
    trials: list[np.ndarray] = []
    labels: list[str] = []
    for _ in range(16):
        noise = 0.05 * rng.normal(size=(8, times.size))
        left = noise.copy()
        left[0] += 3.0 * tone
        right = noise.copy()
        right[1] += 3.0 * tone
        trials.extend([left, right])
        labels.extend(["left", "right"])
    model = DiscriminantGeodesic().fit(
        trials,
        labels,
        sample_rate_hz=160.0,
        window=(0.0, 1.0),
        band=(8.0, 12.0),
    )
    assert model._mode == "tangent"
    predicted = model.predict(trials)
    assert len(predicted) == len(labels)
    assert set(predicted) <= {"left", "right"}


def test_cross_subject_mdm_holds_people_out() -> None:
    rng = np.random.default_rng(3)
    groups = [_subject(f"S{index:03d}", rng) for index in range(1, 5)]
    result = evaluate_cross_subject(
        "mdm",
        groups,
        160.0,
        folds=2,
        inner_folds=2,
        seed=3,
    )
    assert result.metrics["balanced_accuracy"] > 0.7
    assert result.report["chosen"][0]["model"] == "mdm"
    scored = {fold.name for fold in result.folds}
    assert scored == {group.subject for group in groups}


def test_cross_subject_twfb_chooses_inside_the_training_grid() -> None:
    rng = np.random.default_rng(4)
    groups = [_subject(f"S{index:03d}", rng) for index in range(1, 5)]
    search = TwfbSearch(
        population=2,
        generations=1,
        seed=4,
        windows=((0.0, 1.0), (2.0, 3.0)),
        bands=((8.0, 12.0), (13.0, 17.0)),
    )
    result = evaluate_cross_subject(
        "twfb_dgfmdr",
        groups,
        160.0,
        folds=2,
        inner_folds=2,
        seed=4,
        search=search,
    )
    chosen = result.report["chosen"][0]
    assert chosen["window"] in search.windows
    assert chosen["band"] in search.bands
    assert result.metrics["balanced_accuracy"] > 0.5


_MOTOR = ["FC3", "C1", "C3", "C5", "CP3", "FCz", "Cz", "CPz", "C2", "C4", "FC4", "CP4"]


def test_csp_projection_keeps_the_selected_components() -> None:
    covs = np.stack([_spd(np.eye(4)), _spd(2.0 * np.eye(4))])
    filters = np.eye(4)[:, :2]
    projected = _project(covs, filters)
    assert projected.shape == (2, 2, 2)
    assert _distance(projected[0], projected[0]) == 0.0


def test_every_spatial_method_prepares_with_and_without_alignment() -> None:
    rng = np.random.default_rng(6)
    groups = []
    for index in range(2):
        trials = [rng.normal(size=(12, 64)) for _ in range(4)]
        labels = ["left", "right", "left", "right"]
        groups.append(SubjectTrials(f"S{index}", trials, labels, "toy"))
    variants = csp_variants()
    assert len(variants) == len(SPATIAL_METHODS) * 2
    assert {spatial for spatial, _alignment in variants} == set(SPATIAL_METHODS)
    assert {alignment for _spatial, alignment in variants} == {"none", "euclidean"}
    for spatial, alignment in variants:
        prepared, kept = apply_signal(groups, _MOTOR, spatial, alignment)
        assert kept
        assert len(prepared) == 2
        assert prepared[0].trials[0].shape[0] == len(kept)


def test_csp_mdm_chooses_pairs_without_seeing_the_test_subject() -> None:
    rng = np.random.default_rng(7)
    groups = [_subject(f"S{index:03d}", rng) for index in range(1, 5)]
    result = evaluate_csp(
        "mdm",
        groups,
        160.0,
        ["C3", "C4"],
        spatial="none",
        alignment="none",
        pairs=(1,),
        folds=2,
        inner_folds=2,
        seed=7,
    )
    assert result.report["feature"] == "csp"
    assert result.report["spatial"] == "none"
    assert result.report["alignment"] == "none"
    assert result.report["chosen"][0]["csp_pairs"] == 1
    assert result.metrics["balanced_accuracy"] > 0.5
    assert {fold.name for fold in result.folds} == {group.subject for group in groups}


def test_csp_twfb_runs_after_car_and_euclidean_alignment() -> None:
    rng = np.random.default_rng(8)
    groups = [_subject(f"S{index:03d}", rng) for index in range(1, 5)]
    search = TwfbSearch(
        population=2,
        generations=1,
        seed=8,
        windows=((0.0, 1.0), (2.0, 3.0)),
        bands=((8.0, 12.0), (13.0, 17.0)),
    )
    result = evaluate_csp(
        "twfb_dgfmdr",
        groups,
        160.0,
        ["C3", "C4"],
        spatial="car",
        alignment="euclidean",
        pairs=(1,),
        folds=2,
        inner_folds=2,
        seed=8,
        search=search,
    )
    chosen = result.report["chosen"][0]
    assert result.report["spatial"] == "car"
    assert result.report["alignment"] == "euclidean"
    assert chosen["feature"] == "csp"
    assert chosen["csp_pairs"] == 1
    assert chosen["window"] in search.windows
    assert chosen["band"] in search.bands
    assert {fold.name for fold in result.folds} == {group.subject for group in groups}


def test_csp_grid_scores_each_requested_variant() -> None:
    rng = np.random.default_rng(9)
    groups = [_subject(f"S{index:03d}", rng) for index in range(1, 5)]
    results = evaluate_csp_grid(
        "mdm",
        groups,
        160.0,
        ["C3", "C4"],
        variants=(("none", "none"), ("car", "euclidean")),
        pairs=(1,),
        folds=2,
        inner_folds=2,
        seed=9,
    )
    assert [(item.report["spatial"], item.report["alignment"]) for item in results] == [
        ("none", "none"),
        ("car", "euclidean"),
    ]
    assert {item.report["feature"] for item in results} == {"csp"}


def test_tasks_use_one_worker_per_core_and_keep_order() -> None:
    assert _cores() >= 1
    assert _map_tasks([1, 2, 3], lambda item: item * 2) == [2, 4, 6]


def test_csp_grid_writes_the_batch_metrics_and_resumes(tmp_path) -> None:
    rng = np.random.default_rng(10)
    groups = [_subject(f"S{index:03d}", rng) for index in range(1, 5)]
    destination = tmp_path / "paper-csp.json"
    rows = score_csp_grid(
        groups,
        160.0,
        ["C3", "C4"],
        destination,
        dataset="toy",
        models=("mdm",),
        variants=(("none", "none"),),
        pairs=(1,),
        folds=2,
        inner_folds=2,
        seed=10,
    )
    assert len(rows) == 1
    assert rows[0]["features"] == "csp"
    assert rows[0]["classifier"] == "mdm"
    assert rows[0]["evaluation"] == "nested"
    assert rows[0]["metrics"]["balanced_accuracy"] > 0.5
    assert "kappa" in rows[0]["metrics"]
    assert rows[0]["confusion_matrix"]
    table = (tmp_path / "paper-csp-comparison.csv").read_text(encoding="utf-8")
    assert "balanced_accuracy" in table.splitlines()[0]
    assert "kappa" in table.splitlines()[0]
    again = score_csp_grid(
        groups,
        160.0,
        ["C3", "C4"],
        destination,
        dataset="toy",
        models=("mdm",),
        variants=(("none", "none"),),
        pairs=(1,),
        folds=2,
        inner_folds=2,
        seed=10,
    )
    assert len(again) == 1
    stored = comparison_row(
        evaluate_csp(
            "mdm",
            groups,
            160.0,
            ["C3", "C4"],
            pairs=(1,),
            folds=2,
            inner_folds=2,
            seed=10,
        ),
        dataset="toy",
        participants=4,
        seconds=1.0,
    )
    assert stored["features"] == "csp"
    assert stored["spatial"] == "none"
    assert stored["alignment"] == "none"
