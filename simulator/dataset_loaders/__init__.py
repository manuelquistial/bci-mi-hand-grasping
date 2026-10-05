"""
Módulo de cargadores de datasets EEG.
Registro de loaders para uso en SimOpenBCI y detección automática por ruta.
"""
from pathlib import Path
from typing import Any, Type

from .base import BaseDatasetLoader, DatasetMetadata, SubjectRunInfo
from .bci_competition_iv2a import BCICompetitionIV2aLoader
from .liu2024_stroke_mi import Liu2024StrokeMILoader
from .recorded_session import RecordedSessionLoader
from .shuqfa2024 import Shuqfa2024Loader

# Loaders disponibles (orden: más específico primero para auto-detección)
LOADERS: list[Type[BaseDatasetLoader]] = [
    RecordedSessionLoader,
    BCICompetitionIV2aLoader,
    Liu2024StrokeMILoader,
    Shuqfa2024Loader,
]

DATASET_IDS = tuple(L.id for L in LOADERS)


def list_datasets() -> list[dict[str, str]]:
    """Datasets the simulator can play, in registry order."""
    return [{"id": loader.id, "label": loader.label} for loader in LOADERS]


def get_loader_class_by_id(loader_id: str) -> Type[BaseDatasetLoader] | None:
    """Return the loader class for a dataset id, or None."""
    return next((L for L in LOADERS if L.id == loader_id), None)


def create_loader(
    dataset_id: str | None = None,
    file_path: str | Path | None = None,
    **overrides: Any,
) -> BaseDatasetLoader:
    """
    Create a loader instance.

    - If dataset_id is set, instantiate that loader with overrides
      (n_channels, sampling_rate, stream_name, ...).
    - Else if file_path is set, auto-detect from path (overrides applied).
    """
    cls: Type[BaseDatasetLoader]
    if dataset_id:
        found = get_loader_class_by_id(dataset_id)
        if found is None:
            raise ValueError(f"Unknown dataset id: {dataset_id!r}")
        cls = found
    elif file_path is not None:
        path = Path(file_path)
        found = next((L for L in LOADERS if L.detect_from_path(path)), None)
        if found is None:
            raise ValueError(f"No dataset loader matches {path}")
        cls = found
    else:
        raise ValueError("dataset_id or file_path is required")
    return cls(**overrides) if overrides else cls()


def get_loader_for_path(file_path: str | Path, **overrides: Any) -> BaseDatasetLoader:
    """
    Devuelve el primer loader que detecte el path (p. ej. BCI_Competition_IV2a en la ruta).
    """
    return create_loader(file_path=file_path, **overrides)


def get_loader_by_id(loader_id: str, **overrides: Any) -> BaseDatasetLoader | None:
    """Obtiene una instancia del loader por id ('bci_iv2a', 'shuqfa2024')."""
    if get_loader_class_by_id(loader_id) is None:
        return None
    return create_loader(dataset_id=loader_id, **overrides)


def list_subjects_for_dataset(
    dataset_id: str,
    data_root: str | Path,
) -> list[SubjectRunInfo]:
    """Lista sujetos disponibles para un dataset (bci_iv2a, etc.)."""
    loader_cls = get_loader_class_by_id(dataset_id)
    if loader_cls is None:
        return []
    return loader_cls.list_subjects(data_root)


def resolve_data_root(
    dataset_id: str | None,
    data_root: str | Path | None = None,
) -> Path | None:
    """Resolve data_root from explicit path or loader default_data_root()."""
    if data_root is not None:
        return Path(data_root)
    if dataset_id:
        cls = get_loader_class_by_id(dataset_id)
        if cls is not None:
            return cls.default_data_root()
    return None


__all__ = [
    "BaseDatasetLoader",
    "DatasetMetadata",
    "SubjectRunInfo",
    "RecordedSessionLoader",
    "BCICompetitionIV2aLoader",
    "Liu2024StrokeMILoader",
    "Shuqfa2024Loader",
    "LOADERS",
    "DATASET_IDS",
    "list_datasets",
    "create_loader",
    "get_loader_class_by_id",
    "get_loader_for_path",
    "get_loader_by_id",
    "list_subjects_for_dataset",
    "resolve_data_root",
]
