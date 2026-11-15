"""
Base interface for EEG dataset loaders.

New formats (BCI Competition, OpenBMI, etc.) can be added without changing
the simulator. Multi-subject datasets implement ``list_subjects`` and
``resolve_subject_file``.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SubjectRunInfo:
    """A subject and the runs/sessions available for it."""
    subject_id: str
    runs: list[str]  # e.g. ["T", "E"]


@dataclass
class DatasetMetadata:
    """Dataset metadata: channels, sampling rate, stream name."""
    name: str
    n_channels: int
    sampling_rate: float
    channel_names: list[str] | None = None  # optional, used for the LSL description

    def __post_init__(self) -> None:
        if self.channel_names is None:
            self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]


class BaseDatasetLoader(ABC):
    """
    Abstract EEG dataset loader.
    Each concrete implementation reads one format (Shuqfa 2024, Liu 2024, etc.).
    """

    # Short identifier for registry and detection (e.g. "shuqfa2024")
    id: str = "base"
    label: str = "Dataset"

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata (channels, sampling rate, name)."""
        pass

    @abstractmethod
    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        """
        Iterate over the samples in the file.
        Each sample is a list of n_channels floats (one value per channel).
        """
        pass

    def can_handle(self, file_path: str | Path) -> bool:
        """
        Whether this loader can handle the file (by path, extension, etc.).
        Defaults to True; subclasses may restrict by path or content.
        """
        return True

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        """
        Detect the dataset from the path (folder or file name),
        e.g. ``BCI_Competition_IV2a`` in the path.
        """
        return False

    @classmethod
    def default_data_root(cls) -> Path | None:
        """
        Default relative data root under cwd for --subject / --list-subjects.
        Override in multi-subject loaders. Returns None if not applicable.
        """
        return None

    # -------------------------------------------------------------------------
    # Multi-subject datasets (optional; override in loaders with several subjects)
    # -------------------------------------------------------------------------

    @classmethod
    def list_subjects(cls, data_root: str | Path) -> list[SubjectRunInfo]:
        """
        List the subjects and runs available under data_root.
        Defaults to []; multi-subject loaders override this.
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
        Resolve the file path for a subject and, optionally, a run.
        Defaults to None (dataset without a notion of subjects).
        """
        return None


def standard_data_root(dataset_id: str) -> Path:
    """On-disk root for a dataset: ``data/<dataset_id>/`` under the working directory."""
    return Path.cwd() / "data" / dataset_id
