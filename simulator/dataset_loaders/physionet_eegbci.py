"""
Loader para PhysioNet EEGBCI (EEG Motor Movement/Imagery) exportado a TXT.

Formato TXT esperado: una línea por muestra, 8 valores separados por coma (8 canales).
Sin cabecera. Frecuencia: 160 Hz. Coincide con el dataset usado en articulo_1 (preset physionet_eegbci).
Archivos exportados por articulo_1: scripts/export_physionet_to_simulator.py → S001.txt, S002.txt, ...
"""

import re
from pathlib import Path
from typing import Iterator

from .base import BaseDatasetLoader, DatasetMetadata, SubjectRunInfo
from .parsing import iter_float_rows


class PhysioNetEEGBCILoader(BaseDatasetLoader):
    """
    TXT exportado desde PhysioNet EEGBCI: 8 canales, 160 Hz.
    Una línea por muestra; 8 columnas (ch1..ch8). Sin columna de tiempo.
    """

    id = "physionet_eegbci"
    label = "PhysioNet EEG Motor Imagery"

    def __init__(
        self,
        n_channels: int = 8,
        sampling_rate: float = 160.0,
        stream_name: str = "PhysioNet_EEG",
    ) -> None:
        self._metadata = DatasetMetadata(
            name=stream_name,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
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

    def can_handle(self, file_path: str | Path) -> bool:
        path = Path(file_path)
        if path.suffix.lower() != ".txt":
            return False
        return self.detect_from_path(file_path) or self._is_physionet_export(path)

    @classmethod
    def _is_physionet_export(cls, path: Path) -> bool:
        """Detect by path containing physionet_eegbci or filename S001.txt pattern."""
        path_str = path.resolve().as_posix()
        if "physionet_eegbci" in path_str:
            return True
        return bool(re.match(r"^S\d{3}\.txt$", path.name, re.IGNORECASE))

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        path = Path(file_path).resolve()
        path_str = path.as_posix()
        return "physionet_eegbci" in path_str or cls._is_physionet_export(path)

    @classmethod
    def default_data_root(cls) -> Path | None:
        candidates = [
            Path.cwd() / "data" / "physionet_eegbci",
            Path.cwd() / "sample_data" / "physionet_eegbci",
        ]
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]

    # Archivos exportados: S001.txt, S002.txt, ... (sujeto = S + 3 dígitos)
    _SUBJECT_FILE_PATTERN = re.compile(r"^S(\d{3})\.txt$", re.IGNORECASE)

    @classmethod
    def list_subjects(cls, data_root: str | Path) -> list[SubjectRunInfo]:
        """Lista sujetos (S001, S002, ...) con un único 'run' por archivo."""
        root = Path(data_root)
        if not root.is_dir():
            return []
        result = []
        for p in sorted(root.glob("S*.txt")):
            m = cls._SUBJECT_FILE_PATTERN.match(p.name)
            if m:
                result.append(SubjectRunInfo(subject_id=f"S{m.group(1)}", runs=["data"]))
        return result

    @classmethod
    def resolve_subject_file(
        cls,
        data_root: str | Path,
        subject_id: str,
        run: str | None = None,
    ) -> Path | None:
        """Resuelve S001 -> S001.txt; acepta '1' o 'S001'."""
        root = Path(data_root)
        sid = subject_id.strip().upper()
        if not sid.startswith("S"):
            sid = f"S{sid.zfill(3)}"
        elif len(sid) == 2:  # S1
            sid = f"S{sid[1:].zfill(3)}"
        path = root / f"{sid}.txt"
        return path if path.exists() else None
