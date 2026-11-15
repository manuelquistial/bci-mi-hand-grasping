"""Freeze and reload the offline CSP + one or more linear classifiers.

Supported heads: ``logistic_regression`` and ``linear_svm`` — the same
estimators built by :func:`pipeline.classify.models.build_model`.

Shared front-end (filter + CSP) runs once per window; every head scores the
same log-variance features with its own scaler and classifier.

This module is a thin facade; implementations live in ``model_types``,
``model_training`` and ``model_io``.
"""

from __future__ import annotations

from pipeline.online.model_io import load_frozen_model, save_frozen_model
from pipeline.online.model_training import train_frozen_model
from pipeline.online.model_types import (
    ONLINE_CLASSIFIERS,
    FrozenClassifierHead,
    FrozenOnlineModel,
    parse_classifier_names,
)

__all__ = [
    "ONLINE_CLASSIFIERS",
    "FrozenClassifierHead",
    "FrozenOnlineModel",
    "load_frozen_model",
    "parse_classifier_names",
    "save_frozen_model",
    "train_frozen_model",
]
