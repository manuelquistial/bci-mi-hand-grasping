"""
Interfaz base para cargadores de datasets EEG.
Permite añadir nuevos formatos (BCI Competition, OpenBMI, etc.) sin cambiar el simulador.
Soporta datasets con múltiples sujetos mediante list_subjects / resolve_subject_file.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SubjectRunInfo:
    """Información de un sujeto y sus runs/sesiones disponibles."""
    subject_id: str
    runs: list[str]  # ej. ["T", "E"]


@dataclass
class DatasetMetadata:
    """Metadatos del dataset: canales, tasa de muestreo, nombre del stream."""
    name: str
    n_channels: int
    sampling_rate: float
    channel_names: list[str] | None = None  # opcional, para LSL desc

    def __post_init__(self) -> None:
        if self.channel_names is None:
            self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]


class BaseDatasetLoader(ABC):
    """
    Cargador abstracto de datasets EEG.
    Cada implementación concreta sabe leer un formato (Shuqfa 2024, Liu 2024, etc.).
    """

    # Identificador corto para registro y detección (ej: "shuqfa2024")
    id: str = "base"
    label: str = "Dataset"

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Metadatos del dataset (canales, sampling rate, nombre)."""
        pass

    @abstractmethod
    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        """
        Itera sobre las muestras del archivo.
        Cada muestra es una lista de n_channels floats (un valor por canal).
        """
        pass

    def can_handle(self, file_path: str | Path) -> bool:
        """
        Indica si este loader puede manejar el archivo (por ruta, extensión, etc.).
        Por defecto devuelve True; las subclases pueden restringir por path o contenido.
        """
        return True

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        """
        Detección por ruta (carpeta/nombre). Útil para BCI_Competition_IV2a en el path.
        """
        return False

    @classmethod
    def default_data_root(cls) -> Path | None:
        """
        Default relative data root under cwd for --subject / --list-subjects.
        Override in multi-subject loaders. Returns None if not applicable.
        """
        return None


def standard_data_root(dataset_id: str) -> Path:
    """On-disk root for a dataset: ``data/<dataset_id>/`` under the working directory."""
    return Path.cwd() / "data" / dataset_id

    # -------------------------------------------------------------------------
    # Multi-subject datasets (opcional; override en loaders con varios sujetos)
    # -------------------------------------------------------------------------

    @classmethod
    def list_subjects(cls, data_root: str | Path) -> list[SubjectRunInfo]:
        """
        Lista sujetos y sus runs disponibles en data_root.
        Por defecto devuelve []; los loaders con múltiples sujetos lo redefinen.
        """
        return []

    @classmethod
    def resolve_subject_file(
        cls,
        data_root: str | Path,
        subject_id: str,
        run: str | None = None,
    ) -> Path | None:
        """
        Resuelve la ruta del archivo para un sujeto y opcionalmente un run.
        Por defecto devuelve None (dataset sin noción de sujetos).
        """
        return None
