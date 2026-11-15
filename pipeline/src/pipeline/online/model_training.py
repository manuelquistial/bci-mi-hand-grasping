"""Fit shared CSP once and freeze one scaler + classifier per head."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from pipeline.alignment.euclidean import align_trials
from pipeline.classify.models import build_model, scales_features
from pipeline.features.csp import log_variances, spatial_filters
from pipeline.online.model_types import (
    FrozenClassifierHead,
    FrozenOnlineModel,
    parse_classifier_names,
)
from pipeline.preprocess import FilterSpec, apply_notch_then_bandpass


def train_frozen_model(
    trials: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    sample_rate_hz: float,
    classifier_name: str | None = None,
    classifier_names: Sequence[str] | str | None = None,
    filter: FilterSpec | None = None,
    csp_pairs: int = 2,
    apply_filter: bool = True,
    apply_training_ea: bool = True,
    regularization_c: float = 1.0,
    control_classifier: str | None = None,
) -> FrozenOnlineModel:
    """Fit shared CSP once, then freeze one scaler+classifier per requested head.

    Each trial is ``(channels, samples)``. When ``apply_filter`` is true the
    offline notch→bandpass is applied per trial (``filtfilt``). Training EA
    uses the mean covariance of the filtered training trials; online sessions
    recompute EA from rest instead of reusing that matrix.
    """
    names = parse_classifier_names(classifier_names, classifier_name=classifier_name)
    if len(trials) != len(labels) or not trials:
        raise ValueError("Need a labelled trial for every sample")
    spec = filter or FilterSpec()
    prepared: list[np.ndarray] = []
    for trial in trials:
        array = np.asarray(trial, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Each trial must be shaped (channels, samples)")
        if apply_filter:
            filtered = apply_notch_then_bandpass(array.T, sample_rate_hz, spec).T
        else:
            filtered = array
        prepared.append(filtered)
    if apply_training_ea:
        prepared, _weights = align_trials(prepared)
    filters = spatial_filters(
        [trial for trial, label in zip(prepared, labels) if label == "left"],
        [trial for trial, label in zip(prepared, labels) if label == "right"],
        pairs=csp_pairs,
    )
    features = np.stack([log_variances(trial, filters) for trial in prepared], axis=0)
    y = np.asarray([str(label) for label in labels])
    from pipeline.params import ModelParams

    heads: list[FrozenClassifierHead] = []
    for name in names:
        scaler = StandardScaler()
        if scales_features(name):
            learned = scaler.fit_transform(features)
        else:
            learned = features
            scaler.mean_ = np.zeros(features.shape[1])
            scaler.scale_ = np.ones(features.shape[1])
        model = build_model(
            name,
            len(labels),
            ModelParams(regularization_c=regularization_c, csp_pairs=csp_pairs),
        )
        model.fit(learned, y)
        predicted = model.predict(learned)
        accuracy = float(np.mean(predicted == y))
        heads.append(
            FrozenClassifierHead(
                name=name,
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                classifier=model,
                class_labels=tuple(str(item) for item in getattr(model, "classes_", [])),
                offline_accuracy=accuracy,
            )
        )
    return FrozenOnlineModel(
        sample_rate_hz=float(sample_rate_hz),
        filter=spec,
        csp_filters=np.asarray(filters, dtype=np.float64),
        heads=tuple(heads),
        channel_count=int(prepared[0].shape[0]),
        csp_pairs=int(csp_pairs),
        meta={
            "n_train": len(labels),
            "training_ea": apply_training_ea,
            "classifiers": list(names),
        },
        control_classifier=control_classifier or names[0],
    )
