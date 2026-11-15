"""Offline mode specs: dataclass plus JSON loading for registry exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.online.model_types import parse_classifier_names
from pipeline.preprocess import FilterSpec


@dataclass(frozen=True)
class OfflineModeSpec:
    """One offline training mode the user asks to export for online use.

    Training is cross-subject by default (all participants). ``online_subject``
    (or legacy ``subject``) is the participant used later for online
    verification replay and, by default, is held out of training.
    """

    name: str
    dataset: str
    subject: str | None = None
    online_subject: str | None = None
    session_dir: str | None = None
    classifiers: tuple[str, ...] = ("linear_svm", "logistic_regression")
    control_classifier: str | None = None
    alignment: str = "euclidean"
    features: str = "csp"
    cross_subject: bool = True
    hold_out_online_subject: bool = True
    notch_hz: float = 50.0
    band_low_hz: float = 8.0
    band_high_hz: float = 30.0
    order: int = 4
    notch_q: float = 30.0
    csp_pairs: int = 2
    regularization_c: float = 1.0

    def verification_subject(self) -> str | None:
        return self.online_subject or self.subject

    def filter_spec(self) -> FilterSpec:
        return FilterSpec(
            notch_hz=self.notch_hz,
            notch_q=self.notch_q,
            band_low_hz=self.band_low_hz,
            band_high_hz=self.band_high_hz,
            order=self.order,
        )

    def resolved_classifiers(self) -> tuple[str, ...]:
        return parse_classifier_names(self.classifiers)


def load_offline_modes(path: str | Path) -> list[OfflineModeSpec]:
    """Load one or more offline mode specs from JSON or YAML-like JSON.

    Expected shape::

        {
          "modes": [
            {
              "name": "shuqfa-svm-lr",
              "dataset": "shuqfa2024",
              "subject": "S01",
              "classifiers": ["linear_svm", "logistic_regression"]
            }
          ]
        }

    A bare list of mode objects is also accepted.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)
    if isinstance(raw, dict) and "modes" in raw:
        items = raw["modes"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        raise ValueError("Offline mode config must be an object, a list, or {modes: [...]}")
    modes: list[OfflineModeSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each offline mode must be an object")
        modes.append(mode_spec_from_dict(item))
    if not modes:
        raise ValueError("No offline modes found in config")
    return modes


def mode_spec_from_dict(raw: dict[str, Any]) -> OfflineModeSpec:
    name = str(raw.get("name") or "").strip()
    dataset = str(raw.get("dataset") or "").strip()
    if not name:
        raise ValueError("Offline mode needs a name")
    if not dataset:
        raise ValueError(f"Offline mode {name!r} needs a dataset")
    classifiers = raw.get("classifiers") or raw.get("classifier") or (
        "linear_svm",
        "logistic_regression",
    )
    if isinstance(classifiers, str):
        names = parse_classifier_names(classifiers)
    else:
        names = parse_classifier_names([str(item) for item in classifiers])
    filt = raw.get("filter") if isinstance(raw.get("filter"), dict) else {}
    online = raw.get("online_subject")
    if online in (None, ""):
        online = raw.get("subject")
    return OfflineModeSpec(
        name=name,
        dataset=dataset,
        subject=(None if raw.get("subject") in (None, "") else str(raw.get("subject"))),
        online_subject=(None if online in (None, "") else str(online)),
        session_dir=(
            None if raw.get("session_dir") in (None, "") else str(raw.get("session_dir"))
        ),
        classifiers=names,
        control_classifier=(
            None
            if raw.get("control_classifier") in (None, "")
            else str(raw.get("control_classifier"))
        ),
        alignment=str(raw.get("alignment") or "euclidean"),
        features=str(raw.get("features") or "csp"),
        cross_subject=bool(raw.get("cross_subject", True)),
        hold_out_online_subject=bool(raw.get("hold_out_online_subject", True)),
        notch_hz=float(filt.get("notch_hz", raw.get("notch_hz", 50.0))),
        notch_q=float(filt.get("notch_q", raw.get("notch_q", 30.0))),
        band_low_hz=float(filt.get("band_low_hz", raw.get("band_low_hz", 8.0))),
        band_high_hz=float(filt.get("band_high_hz", raw.get("band_high_hz", 30.0))),
        order=int(filt.get("order", raw.get("order", 4))),
        csp_pairs=int(raw.get("csp_pairs", 2)),
        regularization_c=float(raw.get("regularization_c", 1.0)),
    )
