"""Frozen online model packs for the game registry picker."""

from __future__ import annotations

from typing import Any


def list_online_models(*, loso_only: bool = False) -> dict[str, Any]:
    """List frozen packs in ``online-models/`` for the game registry picker."""
    from pipeline.online.registry import OnlineModelRegistry

    models: list[dict[str, Any]] = []
    for entry in OnlineModelRegistry().list_entries():
        offline = dict(entry.offline or {})
        held_out = bool(
            offline.get("held_out_online_subject")
            or offline.get("hold_out_online_subject")
        )
        if loso_only and not held_out:
            continue
        online_subject = str(
            offline.get("online_subject") or offline.get("subject") or ""
        )
        n_train = offline.get("n_train_subjects")
        head = entry.control_classifier or (
            entry.classifiers[0] if entry.classifiers else ""
        )
        head_short = {
            "linear_svm": "SVM",
            "logistic_regression": "LR",
        }.get(str(head), str(head) or "?")
        tag = "LOSO" if held_out else "pooled"
        subject_bit = online_subject or "-"
        train_bit = f", n_train={n_train}" if n_train is not None else ""
        models.append(
            {
                "id": entry.id,
                "name": entry.name,
                "label": f"{head_short} · {tag} {subject_bit}{train_bit}",
                "held_out": held_out,
                "online_subject": online_subject,
                "dataset": str(offline.get("dataset") or ""),
                "control_classifier": str(head),
                "n_train_subjects": n_train,
            }
        )
    return {"models": models}
