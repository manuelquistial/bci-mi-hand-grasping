"""Persist offline-trained frozen models for later online sessions.

Layout::

    pipeline/online-models/
      index.json
      <run_id>/
        manifest.json
        model.joblib

Each registered pack stores the offline recipe (dataset, subject, filter,
classifiers) plus the frozen artifact the online session loads.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import default_registry_root
from pipeline.online.model import (
    FrozenOnlineModel,
    load_frozen_model,
    save_frozen_model,
)
from pipeline.online.registry_modes import (
    OfflineModeSpec,
    load_offline_modes,
    mode_spec_from_dict,
)

__all__ = [
    "OfflineModeSpec",
    "OnlineModelRegistry",
    "RegistryEntry",
    "export_offline_modes",
    "load_offline_modes",
    "mode_spec_from_dict",
]

_ID_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass
class RegistryEntry:
    """One registered offline→online pack."""

    id: str
    name: str
    created_at: str
    offline: dict[str, Any]
    offline_accuracies: dict[str, float | None]
    channel_count: int
    sample_rate_hz: float
    control_classifier: str
    classifiers: list[str]
    model_file: str = "model.joblib"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "offline": self.offline,
            "offline_accuracies": self.offline_accuracies,
            "channel_count": self.channel_count,
            "sample_rate_hz": self.sample_rate_hz,
            "control_classifier": self.control_classifier,
            "classifiers": list(self.classifiers),
            "model_file": self.model_file,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RegistryEntry":
        return cls(
            id=str(raw["id"]),
            name=str(raw.get("name") or raw["id"]),
            created_at=str(raw.get("created_at") or ""),
            offline=dict(raw.get("offline") or {}),
            offline_accuracies={
                str(key): (None if value is None else float(value))
                for key, value in dict(raw.get("offline_accuracies") or {}).items()
            },
            channel_count=int(raw.get("channel_count") or 0),
            sample_rate_hz=float(raw.get("sample_rate_hz") or 0.0),
            control_classifier=str(raw.get("control_classifier") or ""),
            classifiers=[str(item) for item in raw.get("classifiers") or []],
            model_file=str(raw.get("model_file") or "model.joblib"),
            meta=dict(raw.get("meta") or {}),
        )


class OnlineModelRegistry:
    """Filesystem registry of offline exports usable by online-sim."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_registry_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def list_entries(self) -> list[RegistryEntry]:
        entries = [entry for entry, _path in self._scan()]
        entries.sort(key=lambda item: item.created_at, reverse=True)
        return entries

    def get(self, run_id: str) -> RegistryEntry:
        matches = [
            entry
            for entry, _path in self._scan()
            if entry.id == run_id or entry.name == run_id
        ]
        if not matches:
            raise FileNotFoundError(f"No registered offline model named {run_id!r}")
        matches.sort(key=lambda item: item.created_at, reverse=True)
        return matches[0]

    def resolve_dir(self, run_id: str) -> Path:
        entry = self.get(run_id)
        for item, path in self._scan():
            if item.id == entry.id:
                return path
        raise FileNotFoundError(f"No registered offline model named {run_id!r}")

    def load_model(self, run_id: str) -> FrozenOnlineModel:
        folder = self.resolve_dir(run_id)
        entry = RegistryEntry.from_dict(
            json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        )
        return load_frozen_model(folder / entry.model_file)

    def latest(
        self,
        *,
        name: str | None = None,
        dataset: str | None = None,
        subject: str | None = None,
    ) -> RegistryEntry:
        matches = self.list_entries()
        if name:
            matches = [item for item in matches if item.name == name or item.id == name]
        if dataset:
            matches = [
                item for item in matches if str(item.offline.get("dataset") or "") == dataset
            ]
        if subject is not None:
            matches = [
                item
                for item in matches
                if str(item.offline.get("online_subject") or item.offline.get("subject") or "")
                == str(subject)
            ]
        if not matches:
            raise FileNotFoundError("No registered offline model matches the filters")
        return matches[0]

    def register(
        self,
        model: FrozenOnlineModel,
        spec: OfflineModeSpec,
        *,
        run_id: str | None = None,
        replace: bool = False,
    ) -> RegistryEntry:
        """Write ``model.joblib`` + ``manifest.json`` and refresh the index."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = _slug(spec.name)
        chosen_id = run_id or f"{stamp}-{slug}"
        folder = self.root / chosen_id
        if folder.exists() and not replace:
            raise FileExistsError(f"Registry entry already exists: {chosen_id}")
        if folder.exists() and replace:
            for child in folder.iterdir():
                child.unlink()
        folder.mkdir(parents=True, exist_ok=True)
        save_frozen_model(model, folder / "model.joblib")
        entry = RegistryEntry(
            id=chosen_id,
            name=spec.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            offline={
                "dataset": spec.dataset,
                "subject": spec.verification_subject() or "",
                "online_subject": spec.verification_subject() or "",
                "cross_subject": spec.cross_subject,
                "hold_out_online_subject": spec.hold_out_online_subject,
                "held_out_online_subject": (model.meta or {}).get(
                    "held_out_online_subject"
                ),
                "session_dir": spec.session_dir or "",
                "alignment": spec.alignment,
                "features": spec.features,
                "classifiers": list(spec.resolved_classifiers()),
                "control_classifier": model.control_name,
                "filter": asdict(spec.filter_spec()),
                "csp_pairs": spec.csp_pairs,
                "regularization_c": spec.regularization_c,
                "n_train_subjects": (model.meta or {}).get("n_train_subjects"),
                "train_subjects": (model.meta or {}).get("train_subjects"),
            },
            offline_accuracies=dict(model.offline_accuracies),
            channel_count=model.channel_count,
            sample_rate_hz=model.sample_rate_hz,
            control_classifier=model.control_name,
            classifiers=list(model.classifier_names),
            model_file="model.joblib",
            meta=dict(model.meta or {}),
        )
        (folder / "manifest.json").write_text(
            json.dumps(entry.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        self._rewrite_index()
        return entry

    def _scan(self) -> list[tuple[RegistryEntry, Path]]:
        found: list[tuple[RegistryEntry, Path]] = []
        if not self.root.exists():
            return found
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            manifest = folder / "manifest.json"
            if not manifest.exists():
                continue
            try:
                raw = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict) or "id" not in raw:
                continue
            found.append((RegistryEntry.from_dict(raw), folder))
        return found

    def _rewrite_index(self) -> None:
        entries = [entry.to_dict() for entry, _path in self._scan()]
        entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        self.index_path.write_text(
            json.dumps({"entries": entries}, indent=2) + "\n",
            encoding="utf-8",
        )


def export_offline_modes(
    modes: Sequence[OfflineModeSpec],
    *,
    registry: OnlineModelRegistry | None = None,
    replace: bool = False,
    trial_loader: Any | None = None,
    subject_lister: Any | None = None,
    workers: int = 4,
) -> list[RegistryEntry]:
    """Train each offline mode and register the frozen model for online use."""
    from pipeline.online.dataset_model import train_frozen_from_dataset

    store = registry or OnlineModelRegistry()
    exported: list[RegistryEntry] = []
    for spec in modes:
        model = train_frozen_from_dataset(
            dataset=spec.dataset,
            subject=spec.subject,
            online_subject=spec.verification_subject(),
            session_dir=spec.session_dir,
            classifier_names=spec.resolved_classifiers(),
            control_classifier=spec.control_classifier,
            filter=spec.filter_spec(),
            csp_pairs=spec.csp_pairs,
            regularization_c=spec.regularization_c,
            trial_loader=trial_loader,
            subject_lister=subject_lister,
            cross_subject=spec.cross_subject,
            hold_out_online_subject=spec.hold_out_online_subject,
            workers=workers,
        )
        entry = store.register(model, spec, replace=replace)
        exported.append(entry)
    return exported


def _slug(value: str) -> str:
    cleaned = _ID_SAFE.sub("-", value.strip()).strip("-").lower()
    return cleaned or "mode"
