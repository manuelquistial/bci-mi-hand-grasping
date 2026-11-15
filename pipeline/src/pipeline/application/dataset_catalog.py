"""Dataset and participant listings for the game pickers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pipeline.adapters.simulator.catalog import (
    query_simulator_datasets,
    query_simulator_subjects,
)

DatasetLister = Callable[[], list[dict[str, str]]]
SubjectLister = Callable[[str], list[dict[str, str]]]
RecordingOptions = Callable[[], list[dict[str, str]]]


class DatasetCatalog:
    """Cache simulator datasets and participants. Recordings come from a callable."""

    def __init__(
        self,
        *,
        simulator_project: Path,
        recording_options: RecordingOptions,
        dataset_lister: DatasetLister | None = None,
        subject_lister: SubjectLister | None = None,
    ) -> None:
        self._simulator_project = simulator_project
        self._recording_options = recording_options
        self._dataset_lister = dataset_lister
        self._subject_lister = subject_lister
        self._datasets: list[dict[str, str]] | None = None
        self._subjects: dict[str, list[dict[str, str]]] = {}

    def list_datasets(self) -> dict[str, Any]:
        if self._datasets is None:
            lister = self._dataset_lister or (
                lambda: query_simulator_datasets(self._simulator_project)
            )
            self._datasets = lister()
        datasets: list[dict[str, str]] = [
            {
                "id": item["id"],
                "label": item["label"],
                "dataset": item["id"],
                "sessionDir": "",
            }
            for item in self._datasets
            if item["id"] != "recorded_session"
        ]
        datasets.extend(self._recording_options())
        return {"datasets": datasets}

    def list_subjects(self, dataset: str) -> dict[str, Any]:
        if not dataset or dataset == "recorded_session":
            return {"subjects": []}
        cached = self._subjects.get(dataset)
        if cached is None:
            lister = self._subject_lister or (
                lambda dataset_id: query_simulator_subjects(self._simulator_project, dataset_id)
            )
            cached = lister(dataset)
            self._subjects[dataset] = cached
        return {"subjects": cached}
