"""
Loader para BCI Competition IV Dataset 2a (formato TXT).

Especificaciones: 22 canales EEG, 250 Hz, sin cabecera; primera columna = tiempo, resto = canales.
Sujetos: A01..A09; runs: T (training), E (evaluation). Archivos: A01T.txt, A01E.txt, etc.

Descripción completa del dataset (paradigma, 4 clases MI, eventos, artefactos): ver
docs/bci-iv2a-dataset.md.
"""
import re
from pathlib import Path
from typing import Iterator

from .base import BaseDatasetLoader, DatasetMetadata, SubjectRunInfo
from .parsing import iter_float_rows


# Nombres estándar de canales para BCI Competition IV 2a (22 EEG)
BCI_IV2A_CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]


class BCICompetitionIV2aLoader(BaseDatasetLoader):
    """
    Archivos TXT: una línea por muestra.
    Columna 0 = tiempo (se ignora para el stream), columnas 1..22 = 22 canales EEG.
    Sin líneas de comentario (no hay '%').
    """

    id = "bci_iv2a"
    label = "BCI Competition IV 2a"

    def __init__(
        self,
        n_channels: int = 22,
        sampling_rate: float = 250.0,
        stream_name: str = "BCI_IV2a",
    ) -> None:
        self._metadata = DatasetMetadata(
            name=stream_name,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            channel_names=BCI_IV2A_CHANNEL_NAMES[:n_channels] if n_channels <= 22 else None,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        yield from iter_float_rows(
            file_path,
            n_channels=self._metadata.n_channels,
            start_col=1,
            comment_prefix=None,
        )

    def can_handle(self, file_path: str | Path) -> bool:
        path = Path(file_path)
        return path.suffix.lower() in (".txt",) and self.detect_from_path(file_path)

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        path = Path(file_path).resolve()
        path_str = path.as_posix()
        return "BCI_Competition_IV2a" in path_str or "BCI_Competition_IV2A" in path_str

    @classmethod
    def default_data_root(cls) -> Path | None:
        candidates = [
            Path.cwd() / "data" / "sujetos_BCI_Competition_IV2a",
            Path.cwd() / "sample_data" / "bci_iv2a",
        ]
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]

    # BCI IV 2a: archivos {subject_id}{run}.txt (ej. A01T.txt, A01E.txt)
    _FILE_PATTERN = re.compile(r"^([A-Za-z0-9]+)([A-Za-z])\.txt$", re.IGNORECASE)

    @classmethod
    def list_subjects(cls, data_root: str | Path) -> list[SubjectRunInfo]:
        """Lista todos los sujetos y sus runs (T, E) en la carpeta del dataset."""
        root = Path(data_root)
        if not root.is_dir():
            return []
        subject_runs: dict[str, list[str]] = {}
        for p in root.glob("*.txt"):
            m = cls._FILE_PATTERN.match(p.name)
            if m:
                sid, run = m.group(1), m.group(2).upper()
                subject_runs.setdefault(sid, []).append(run)
        for sid in subject_runs:
            subject_runs[sid] = sorted(set(subject_runs[sid]))
        return [
            SubjectRunInfo(subject_id=sid, runs=runs)
            for sid in sorted(subject_runs.keys())
            for runs in [subject_runs[sid]]
        ]

    @classmethod
    def resolve_subject_file(
        cls,
        data_root: str | Path,
        subject_id: str,
        run: str | None = None,
    ) -> Path | None:
        """Devuelve la ruta del archivo para el sujeto y run (ej. A01, T -> A01T.txt)."""
        root = Path(data_root)
        if run is not None:
            run = run.upper()
        info_list = cls.list_subjects(root)
        subject_ids = {s.subject_id: s for s in info_list}
        if subject_id not in subject_ids:
            return None
        runs = subject_ids[subject_id].runs
        if run and run in runs:
            run_use = run
        elif "T" in runs:
            run_use = "T"
        else:
            run_use = runs[0] if runs else None
        if run_use is None:
            return None
        path = root / f"{subject_id}{run_use}.txt"
        return path if path.exists() else None
