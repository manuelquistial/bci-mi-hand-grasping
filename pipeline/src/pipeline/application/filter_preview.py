"""Pre-model filter preview: one chart participant, a cohort for supervised steps."""

from __future__ import annotations

from typing import Any, Callable

from pipeline.application.cohort_service import CohortService, align_groups, group_named
from pipeline.application.dataset_catalog import DatasetCatalog
from pipeline.application.preview_window import PreviewWindows
from pipeline.alignment import apply_alignment
from pipeline.classify import CLASSIFIERS, SCHEMES, SubjectTrials, evaluate_classifier
from pipeline.features import METHODS as FEATURE_METHODS
from pipeline.features import extract_features
from pipeline.params import ModelParams
from pipeline.preprocess import decimate_time, traces

SUPERVISED_FEATURES = {"tdp", "csp", "fbcsp"}


def feature_chart(
    feature: str,
    groups: list[SubjectTrials],
    example: str,
    rate: float,
    channels: list[str],
    params: ModelParams | None = None,
) -> tuple[list[str], str, list[list[float]]]:
    """Chart one participant. TDP, CSP, and FBCSP are fit on the whole cohort."""
    shown = group_named(groups, example)
    if feature not in SUPERVISED_FEATURES or len(groups) == 1:
        view = extract_features(feature, shown.trials, shown.labels, rate, channels, params)
        detail = view.detail
        if len(groups) > 1:
            detail = f"{detail} · {len(groups)} participants"
        return view.rows, detail, traces(view.series)
    trials = [trial for group in groups for trial in group.trials]
    labels = [label for group in groups for label in group.labels]
    view = extract_features(feature, trials, labels, rate, channels, params)
    start = 0
    rows = view.rows
    series = view.series
    for group in groups:
        end = start + len(group.labels)
        if group.subject == example:
            rows = view.rows[start:end]
            series = view.series[:, start:end]
            break
        start = end
    rest = view.detail.split(" · ", 1)[1] if " · " in view.detail else view.detail
    detail = f"{len(rows)} trials · {rest} · {len(groups)} participants"
    return rows, detail, traces(series)


class FilterPreviewService:
    """Notch, bandpass, spatial, alignment, one descriptor, then a classifier."""

    def __init__(
        self,
        *,
        catalog: DatasetCatalog,
        windows: PreviewWindows,
        cohort: CohortService,
        subject_chooser: Callable[[list[dict[str, str]]], dict[str, str]],
    ) -> None:
        self._catalog = catalog
        self._windows = windows
        self._cohort_service = cohort
        self._subject_chooser = subject_chooser

    def preview_filter(
        self,
        *,
        dataset: str,
        subject: str | None,
        session_dir: str | None,
        notch_hz: float,
        band_low_hz: float,
        band_high_hz: float,
        order: int,
        spatial: str = "none",
        alignment: str = "none",
        features: str = "none",
        classifier: str = "none",
        evaluation: str = "cross_validation",
        params: ModelParams | None = None,
    ) -> dict[str, Any]:
        """Notch, bandpass, spatial, alignment, one descriptor, then a classifier.

        The client does not choose a participant. Charts use one person drawn
        from the dataset. Supervised descriptors and the classifier use every
        participant. Euclidean alignment stays on each participant's own trials.
        """
        del subject
        tuned = params or ModelParams()
        example = self._pick_subject(dataset)
        prepared = self._windows.prepare(
            dataset=dataset,
            subject=example,
            session_dir=session_dir,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            params=tuned,
        )
        rate = prepared["rate"]
        spec = prepared["spec"]
        channels = prepared["channels"]
        spatial_name = prepared["spatial_name"]
        spatial_labels = prepared["spatial_labels"]
        spatial_view = prepared["spatial_view"]
        current = prepared["current"]
        current_channels = prepared["current_channels"]
        raw_view = prepared["raw_view"]
        filtered_view = prepared["filtered_view"]
        seconds = prepared["seconds"]
        alignment_name = (alignment or "none").strip().lower()
        feature_name = (features or "none").strip().lower()
        if feature_name not in FEATURE_METHODS:
            raise ValueError(f"Unknown feature method {features!r}")
        classifier_name = (classifier or "none").strip().lower()
        evaluation_name = (evaluation or "cross_validation").strip().lower()
        if classifier_name not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {classifier!r}")
        if classifier_name != "none" and evaluation_name not in SCHEMES:
            raise ValueError(f"Unknown evaluation {evaluation!r}")
        needs_trials = (
            alignment_name == "euclidean" or feature_name != "none" or classifier_name != "none"
        )
        needs_cohort = classifier_name != "none" or feature_name in SUPERVISED_FEATURES
        groups: list[SubjectTrials] = []
        if needs_trials:
            groups = self._cohort_service.cohort(
                dataset=dataset,
                session_dir=session_dir,
                sample_rate_hz=rate,
                spec=spec,
                spatial=spatial_name,
                channels=current_channels,
                only_subject=None if needs_cohort else example,
                params=tuned,
            )
            shown_name = example or "participant"
            if needs_cohort and shown_name not in {group.subject for group in groups}:
                picked = self._subject_chooser(
                    [{"id": group.subject, "label": group.subject} for group in groups]
                )
                example = None if str(picked["id"]) == "participant" else str(picked["id"])
                prepared = self._windows.prepare(
                    dataset=dataset,
                    subject=example,
                    session_dir=session_dir,
                    notch_hz=notch_hz,
                    band_low_hz=band_low_hz,
                    band_high_hz=band_high_hz,
                    order=order,
                    spatial=spatial,
                    params=tuned,
                )
                current = prepared["current"]
                current_channels = prepared["current_channels"]
                channels = prepared["channels"]
                spatial_labels = prepared["spatial_labels"]
                spatial_view = prepared["spatial_view"]
                raw_view = prepared["raw_view"]
                filtered_view = prepared["filtered_view"]
                seconds = prepared["seconds"]
        if alignment_name == "euclidean":
            groups, weights = align_groups(groups, example or "participant")
            current = apply_alignment(current, weights)
            alignment_view = traces(decimate_time(current))
            shown = group_named(groups, example or "participant")
            alignment_trials = len(shown.trials)
        elif alignment_name != "none":
            raise ValueError(f"Unknown alignment {alignment!r}")
        else:
            alignment_view = []
            alignment_trials = 0
        feature_view: list[list[float]] = []
        feature_rows: list[str] = []
        feature_detail = ""
        if feature_name != "none":
            feature_rows, feature_detail, feature_view = feature_chart(
                feature_name,
                groups,
                example or "participant",
                rate,
                current_channels,
                tuned,
            )
        classifier_detail = ""
        classifier_folds: list[dict[str, float | str]] = []
        classifier_accuracy: float | None = None
        classifier_metrics: dict[str, float | None] = {}
        confusion_labels: list[str] = []
        confusion_matrix: list[list[int]] = []
        if classifier_name != "none":
            if feature_name == "none":
                raise ValueError("Choose a feature method before the classifier")
            scored = evaluate_classifier(
                classifier_name,
                evaluation_name,
                feature_name,
                groups,
                sample_rate_hz=rate,
                channels=current_channels,
                params=tuned,
            )
            classifier_accuracy = scored.accuracy
            classifier_detail = scored.detail
            classifier_folds = [
                {"name": item.name, "accuracy": round(item.accuracy, 4)}
                for item in scored.folds
            ]
            classifier_metrics = {
                key: None if value is None else round(float(value), 4)
                for key, value in scored.metrics.items()
            }
            confusion_labels = scored.labels
            confusion_matrix = scored.matrix
        catalog = self._catalog.list_subjects(dataset)["subjects"]
        participants = len(groups) if groups else max(len(catalog), 1)
        shown_subject = "" if example in (None, "participant") else example
        return {
            "dataset": dataset,
            "subject": shown_subject,
            "participants": participants,
            "sample_rate_hz": rate,
            "seconds": seconds,
            "channels": channels,
            "notch_hz": spec.notch_hz,
            "band_low_hz": spec.band_low_hz,
            "band_high_hz": spec.band_high_hz,
            "order": spec.order,
            "spatial": spatial_name,
            "spatial_channels": spatial_labels,
            "alignment": alignment_name,
            "alignment_trials": alignment_trials,
            "alignment_channels": current_channels if alignment_name == "euclidean" else [],
            "features": feature_name,
            "feature_detail": feature_detail,
            "feature_channels": feature_rows,
            "classifier": classifier_name,
            "evaluation": evaluation_name if classifier_name != "none" else "",
            "accuracy": None if classifier_accuracy is None else round(classifier_accuracy, 4),
            "classifier_detail": classifier_detail,
            "classifier_folds": classifier_folds,
            "metrics": classifier_metrics,
            "confusion_labels": confusion_labels,
            "confusion_matrix": confusion_matrix,
            "raw": traces(raw_view),
            "filtered": traces(filtered_view),
            "spatial_signal": spatial_view,
            "alignment_signal": alignment_view,
            "feature_signal": feature_view,
        }

    def _pick_subject(self, dataset: str) -> str | None:
        listed = list(self._catalog.list_subjects(dataset)["subjects"])
        if not listed:
            return None
        picked = self._subject_chooser(listed)
        return str(picked["id"])

