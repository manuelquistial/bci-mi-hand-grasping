"""
Loader para Liu et al. 2024: imaginación de agarre en pacientes con ACV agudo.

EDF preprocesado (edffile/): 33 señales, 500 Hz, 320 s.
El stream lleva los 29 EEG. CPz (referencia, vale 0), HEOL/HEOR y el canal de marcadores no se emiten.
Los marcadores son impulsos: 1 instrucción, 2 imaginación motora, 3 descanso.
Las clases publicadas alternan izquierda y derecha, empezando por la izquierda.

Ver docs/liu2024-stroke-mi.md.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import BaseDatasetLoader, DatasetMetadata, SubjectRunInfo, standard_data_root
from .edf import EdfHeader, iter_channel, iter_samples, read_header
from .events import TrialEvent

# Orden del EDF. CPz (índice 17) es la referencia; el último canal no tiene nombre
# y lleva los marcadores de etapa.
FILE_CHANNEL_LABELS = (
    "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
    "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CPz", "CP3", "CP4",
    "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2",
    "HEOL", "HEOR", "",
)
EOG_LABELS = frozenset({"HEOL", "HEOR", "VEOL", "VEOR"})
_HANDS = ("left", "right")
_FILE_NAME = re.compile(r"^sub-(\d+)_task-motor-imagery_eeg\.edf$", re.IGNORECASE)
_RUN = "imagery"


def playback_channels(header: EdfHeader) -> list[int]:
    """29 EEG. Sin CPz (referencia), sin EOG y sin el canal de marcadores."""
    eeg: list[int] = []
    for index, signal in enumerate(header.signals):
        label = signal.label.strip()
        if not label or label.upper() == "CPZ" or label.upper() in EOG_LABELS:
            continue
        eeg.append(index)
    return eeg


def events_from_marker(samples: Iterator[float]) -> list[TrialEvent]:
    """Impulsos 1/2/3 → instrucción, inicio de imaginación y descanso."""
    events: list[TrialEvent] = []
    trial = 0
    hand = _HANDS[0]
    for sample_index, value in enumerate(samples):
        code = int(round(value))
        if code == 0:
            continue
        if code == 1:
            hand = _HANDS[trial % 2]
            trial += 1
            events.append(TrialEvent(sample_index, "1", hand, "instruction"))
        elif code == 2:
            events.append(TrialEvent(sample_index, "2", hand, "trial_start"))
        elif code == 3:
            events.append(TrialEvent(sample_index, "3", "rest", "rest"))
    return events


class Liu2024StrokeMILoader(BaseDatasetLoader):
    """EDF de Liu et al. 2024. Stream: 29 EEG, 500 Hz."""

    id = "liu2024_stroke"
    label = "Liu 2024 stroke grasp MI"

    def __init__(
        self,
        n_channels: int | None = None,
        sampling_rate: float | None = None,
        stream_name: str | None = None,
    ) -> None:
        self._channel_override = n_channels
        self._rate_override = sampling_rate
        self._name_override = stream_name
        names = [
            label
            for label in FILE_CHANNEL_LABELS
            if label and label != "CPz" and label not in EOG_LABELS
        ]
        if n_channels is not None:
            names = names[:n_channels]
        self._metadata = DatasetMetadata(
            name=stream_name or "Liu2024_Stroke_MI",
            n_channels=len(names),
            sampling_rate=sampling_rate if sampling_rate is not None else 500.0,
            channel_names=names,
        )
        self._header: EdfHeader | None = None
        self._channels: list[int] | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def bind(self, file_path: str | Path) -> None:
        header = read_header(file_path)
        channels = playback_channels(header)
        if not channels:
            raise ValueError(f"{Path(file_path).name} has no EEG channels to stream")
        if self._channel_override is not None:
            if self._channel_override < 1 or self._channel_override > len(channels):
                raise ValueError(
                    f"{Path(file_path).name} streams {len(channels)} channels, "
                    f"CLI asked for {self._channel_override}"
                )
            channels = channels[: self._channel_override]
        rate = header.sampling_rate
        if self._rate_override is not None and abs(self._rate_override - rate) > 0.1:
            raise ValueError(
                f"{Path(file_path).name} is {rate:g} Hz, CLI asked for {self._rate_override:g} Hz"
            )
        self._header = header
        self._channels = channels
        self._metadata = DatasetMetadata(
            name=self._name_override or "Liu2024_Stroke_MI",
            n_channels=len(channels),
            sampling_rate=rate,
            channel_names=[header.signals[index].label for index in channels],
        )

    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        self.bind(file_path)
        assert self._header is not None and self._channels is not None
        yield from iter_samples(file_path, self._header, self._channels)

    def load_events(self, file_path: str | Path) -> list[TrialEvent]:
        self.bind(file_path)
        assert self._header is not None
        marker = next(
            (index for index, signal in enumerate(self._header.signals) if not signal.label.strip()),
            None,
        )
        if marker is None:
            return []
        return events_from_marker(iter_channel(file_path, self._header, marker))

    def can_handle(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() == ".edf" and self.detect_from_path(file_path)

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        return _FILE_NAME.match(Path(file_path).name) is not None

    @classmethod
    def default_data_root(cls) -> Path | None:
        return standard_data_root(cls.id)

    @classmethod
    def list_subjects(cls, data_root: str | Path) -> list[SubjectRunInfo]:
        root = Path(data_root)
        if not root.is_dir():
            return []
        return [
            SubjectRunInfo(subject_id=subject_id, runs=[_RUN])
            for subject_id, _path in _subject_files(root)
        ]

    @classmethod
    def resolve_subject_file(
        cls,
        data_root: str | Path,
        subject_id: str,
        run: str | None = None,
    ) -> Path | None:
        if run is not None and run.lower() not in {_RUN, "task-motor-imagery"}:
            return None
        wanted = _normalize_subject(subject_id)
        if wanted is None:
            return None
        for subject, path in _subject_files(Path(data_root)):
            if subject == wanted:
                return path
        return None


def _normalize_subject(subject_id: str) -> str | None:
    text = subject_id.strip()
    match = re.fullmatch(r"(?:sub-)?(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return f"sub-{int(match.group(1)):02d}"


def _subject_files(root: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.edf")):
        match = _FILE_NAME.match(path.name)
        if match is None:
            continue
        subject_id = f"sub-{int(match.group(1)):02d}"
        current = found.get(subject_id)
        if current is None or _prefer_edf(path, current):
            found[subject_id] = path
    return sorted(found.items())


def _prefer_edf(candidate: Path, current: Path) -> bool:
    """Prefer the BIDS eeg/ file over another copy of the same subject."""
    candidate_eeg = "eeg" in candidate.parts
    current_eeg = "eeg" in current.parts
    return candidate_eeg and not current_eeg
