"""Trial/event sidecar format for EEG playback.

CSV next to the EEG file: ``{stem}.events.csv``

Columns:
  sample_index,code,label[,kind]

- sample_index: 0-based index into the EEG sample stream
- code: dataset-native code (e.g. T1, 769)
- label: short display label (e.g. left, right, trial)
- kind: optional; ``trial_start`` marks trial onsets (default if omitted for cue codes)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialEvent:
    sample_index: int
    code: str
    label: str
    kind: str = "trial_start"

    @property
    def marker_string(self) -> str:
        """LSL Markers payload (string channel)."""
        return f"{self.kind}:{self.code}:{self.label}"


def events_path_for(eeg_file: str | Path) -> Path:
    path = Path(eeg_file)
    return path.with_suffix(".events.csv")


def load_events_csv(path: str | Path) -> list[TrialEvent]:
    """Load and sort events from a sidecar CSV."""
    file_path = Path(path)
    if not file_path.is_file():
        return []

    events: list[TrialEvent] = []
    with file_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        fields = {f.strip().lower(): f for f in reader.fieldnames}
        if "sample_index" not in fields:
            raise ValueError(
                f"{file_path}: events CSV must include a sample_index column"
            )
        for row in reader:
            if not row or all(not (v or "").strip() for v in row.values()):
                continue
            idx = int(float(row[fields["sample_index"]].strip()))
            code = (row[fields["code"]].strip() if "code" in fields else "") or "event"
            label = (
                row[fields["label"]].strip() if "label" in fields else code
            ) or code
            kind = (
                row[fields["kind"]].strip() if "kind" in fields else "trial_start"
            ) or "trial_start"
            events.append(
                TrialEvent(sample_index=idx, code=code, label=label, kind=kind)
            )
    events.sort(key=lambda e: (e.sample_index, e.code, e.label))
    return events


def resolve_events(
    eeg_file: str | Path,
    events_file: str | Path | None = None,
) -> list[TrialEvent]:
    """Load events from an explicit path or the default sidecar next to the EEG file."""
    if events_file is not None:
        return load_events_csv(events_file)
    sidecar = events_path_for(eeg_file)
    if sidecar.is_file():
        return load_events_csv(sidecar)
    return []
