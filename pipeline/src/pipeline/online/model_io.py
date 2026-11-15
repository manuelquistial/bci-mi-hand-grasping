"""Persist and reload frozen online models (joblib payloads, v1 and v2)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pipeline.online.model_types import (
    ONLINE_CLASSIFIERS,
    FrozenClassifierHead,
    FrozenOnlineModel,
)
from pipeline.preprocess import FilterSpec


def save_frozen_model(model: FrozenOnlineModel, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "sample_rate_hz": model.sample_rate_hz,
        "filter": asdict(model.filter),
        "csp_filters": model.csp_filters,
        "channel_count": model.channel_count,
        "csp_pairs": model.csp_pairs,
        "meta": model.meta or {},
        "control_classifier": model.control_name,
        "heads": [
            {
                "name": head.name,
                "scaler_mean": head.scaler_mean,
                "scaler_scale": head.scaler_scale,
                "classifier": head.classifier,
                "class_labels": list(head.class_labels),
                "offline_accuracy": head.offline_accuracy,
            }
            for head in model.heads
        ],
    }
    joblib.dump(payload, destination)
    return destination


def load_frozen_model(path: str | Path) -> FrozenOnlineModel:
    payload = joblib.load(Path(path))
    raw_filter = payload["filter"]
    spec = FilterSpec(**raw_filter) if isinstance(raw_filter, dict) else raw_filter
    if payload.get("heads"):
        heads = tuple(_head_from_payload(item) for item in payload["heads"])
    else:
        # v1 single-classifier artifacts
        name = str(payload["classifier_name"])
        if name not in ONLINE_CLASSIFIERS:
            raise ValueError(f"Unsupported frozen classifier {name!r}")
        heads = (
            FrozenClassifierHead(
                name=name,
                scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
                classifier=payload["classifier"],
                class_labels=tuple(str(item) for item in payload["class_labels"]),
                offline_accuracy=(
                    None
                    if payload.get("offline_accuracy") is None
                    else float(payload["offline_accuracy"])
                ),
            ),
        )
    return FrozenOnlineModel(
        sample_rate_hz=float(payload["sample_rate_hz"]),
        filter=spec,
        csp_filters=np.asarray(payload["csp_filters"], dtype=np.float64),
        heads=heads,
        channel_count=int(payload["channel_count"]),
        csp_pairs=int(payload["csp_pairs"]),
        meta=dict(payload.get("meta") or {}),
        control_classifier=payload.get("control_classifier") or heads[0].name,
    )


def _head_from_payload(item: dict[str, Any]) -> FrozenClassifierHead:
    name = str(item["name"])
    if name not in ONLINE_CLASSIFIERS:
        raise ValueError(f"Unsupported frozen classifier {name!r}")
    return FrozenClassifierHead(
        name=name,
        scaler_mean=np.asarray(item["scaler_mean"], dtype=np.float64),
        scaler_scale=np.asarray(item["scaler_scale"], dtype=np.float64),
        classifier=item["classifier"],
        class_labels=tuple(str(label) for label in item["class_labels"]),
        offline_accuracy=(
            None if item.get("offline_accuracy") is None else float(item["offline_accuracy"])
        ),
    )
