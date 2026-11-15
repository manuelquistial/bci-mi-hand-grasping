"""BaseDatasetLoader exposes the multi-subject hooks with safe defaults."""
from pathlib import Path

from dataset_loaders.base import BaseDatasetLoader, standard_data_root


def test_base_class_has_subject_hooks(tmp_path: Path) -> None:
    assert "list_subjects" in vars(BaseDatasetLoader)
    assert "resolve_subject_file" in vars(BaseDatasetLoader)
    assert BaseDatasetLoader.list_subjects(tmp_path) == []
    assert BaseDatasetLoader.resolve_subject_file(tmp_path, "1") is None
    assert BaseDatasetLoader.resolve_subject_file(tmp_path, "1", "T") is None


def test_standard_data_root_is_module_level(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert standard_data_root("x") == tmp_path / "data" / "x"
    assert not hasattr(BaseDatasetLoader, "standard_data_root")
