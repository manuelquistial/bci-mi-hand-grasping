"""
Loader para Shuqfa et al. 2024: PhysioNet EEGBCI depurado.

103 sujetos, 12 registros de tarea (sin las líneas base). Cada registro es un CSV
de 64 canales a 160 Hz. Las anotaciones traen la etiqueta y el índice de inicio
de cada periodo; la duración no es fija.

Registros 02, 06 y 10: imaginación unilateral. Etiquetas 4 reposo, 5 izquierda, 6 derecha.

Ver docs/shuqfa2024.md.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import BaseDatasetLoader, DatasetMetadata, SubjectRunInfo, standard_data_root
from .events import TrialEvent
from .parsing import iter_float_rows

# Orden 10-10 de EEGBCI / BCI2000, 64 canales.
CHANNEL_NAMES = (
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1", "Oz", "O2", "Iz",
)
# Código depurado -> (etiqueta del stream, kind).
CLASS_LABELS = {
    1: ("rest", "rest"),
    2: ("left", "trial_start"),
    3: ("right", "trial_start"),
    4: ("rest", "rest"),
    5: ("left", "trial_start"),
    6: ("right", "trial_start"),
    7: ("rest", "rest"),
    8: ("both_fists", "trial_start"),
    9: ("both_feet", "trial_start"),
    10: ("rest", "rest"),
    11: ("both_fists", "trial_start"),
    12: ("both_feet", "trial_start"),
}
# Primer registro de imaginación unilateral cuando no se pide otro.
DEFAULT_RUN = "02"
_SIG_NAME = re.compile(r"^SUB_(\d+)_SIG_(\d+)\.csv$", re.IGNORECASE)


def events_from_annotations(path: Path) -> list[TrialEvent]:
    """Cada fila: etiqueta, duración (s), muestras, inicio, fin. Inicio en base 1."""
    events: list[TrialEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [item.strip() for item in line.split(",") if item.strip()]
        if len(parts) < 4:
            continue
        try:
            code = int(float(parts[0]))
            onset = int(float(parts[3])) - 1
        except ValueError:
            continue
        if onset < 0:
            continue
        label, kind = CLASS_LABELS.get(code, (str(code), "trial_start"))
        events.append(TrialEvent(onset, str(code), label, kind))
    events.sort(key=lambda event: (event.sample_index, event.code))
    return events


class Shuqfa2024Loader(BaseDatasetLoader):
    """Un CSV de Shuqfa 2024. Stream: 64 EEG, 160 Hz. Run = 01 … 12."""

    id = "shuqfa2024"
    label = "Shuqfa 2024 curated PhysioNet"

    def __init__(
        self,
        n_channels: int | None = None,
        sampling_rate: float | None = None,
        stream_name: str | None = None,
    ) -> None:
        self._channel_override = n_channels
        self._rate_override = sampling_rate
        self._name_override = stream_name
        count = n_channels if n_channels is not None else len(CHANNEL_NAMES)
        if count < 1 or count > len(CHANNEL_NAMES):
            raise ValueError(
                f"shuqfa2024 has {len(CHANNEL_NAMES)} channels, CLI asked for {count}"
            )
        rate = 160.0 if sampling_rate is None else sampling_rate
        self._metadata = DatasetMetadata(
            name=stream_name or "Shuqfa2024_EEG",
            n_channels=count,
            sampling_rate=rate,
            channel_names=list(CHANNEL_NAMES[:count]),
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        yield from iter_float_rows(
            file_path,
            n_channels=self._metadata.n_channels,
            start_col=0,
            comment_prefix="%",
        )

    def load_events(self, file_path: str | Path) -> list[TrialEvent]:
        annotations = annotation_file_for(Path(file_path))
        if annotations is None:
            return []
        return events_from_annotations(annotations)

    def can_handle(self, file_path: str | Path) -> bool:
        return self.detect_from_path(file_path)

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        return _SIG_NAME.match(Path(file_path).name) is not None

    @classmethod
    def default_data_root(cls) -> Path | None:
        return standard_data_root(cls.id)

    @classmethod
    def list_subjects(cls, data_root: str | Path) -> list[SubjectRunInfo]:
        grouped: dict[str, list[str]] = {}
        for subject_id, run, _path in _signal_files(Path(data_root)):
            grouped.setdefault(subject_id, []).append(run)
        return [
            SubjectRunInfo(subject_id=subject_id, runs=runs)
            for subject_id, runs in sorted(grouped.items())
        ]

    @classmethod
    def resolve_subject_file(
        cls,
        data_root: str | Path,
        subject_id: str,
        run: str | None = None,
    ) -> Path | None:
        wanted = _normalize_subject(subject_id)
        if wanted is None:
            return None
        requested = _normalize_run(run) if run is not None else None
        if run is not None and requested is None:
            return None
        matches = [
            (item_run, path)
            for item_subject, item_run, path in _signal_files(Path(data_root))
            if item_subject == wanted
        ]
        if not matches:
            return None
        matches.sort()
        if requested is None:
            for item_run, path in matches:
                if item_run == DEFAULT_RUN:
                    return path
            return matches[0][1]
        for item_run, path in matches:
            if item_run == requested:
                return path
        return None


def annotation_file_for(signal_file: Path) -> Path | None:
    name = re.sub(r"_SIG_", "_ANN_", signal_file.name, count=1, flags=re.IGNORECASE)
    candidate = signal_file.with_name(name)
    return candidate if candidate.is_file() else None


def _normalize_subject(subject_id: str) -> str | None:
    match = re.fullmatch(r"(?:sub_|s)?(\d+)", subject_id.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    return f"S{int(match.group(1)):03d}"


def _normalize_run(run: str) -> str | None:
    match = re.fullmatch(r"(?:run-?)?(\d+)", run.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    number = int(match.group(1))
    if number < 1 or number > 12:
        return None
    return f"{number:02d}"


def _signal_files(root: Path) -> list[tuple[str, str, Path]]:
    if not root.is_dir():
        return []
    found: list[tuple[str, str, Path]] = []
    for path in sorted(root.rglob("*.csv")):
        match = _SIG_NAME.match(path.name)
        if match is None:
            continue
        subject_id = f"S{int(match.group(1)):03d}"
        run = f"{int(match.group(2)):02d}"
        found.append((subject_id, run, path))
    return found
