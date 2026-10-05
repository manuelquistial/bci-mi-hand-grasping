"""Tests for loader registry and multi-subject helpers."""
from pathlib import Path

import pytest

from dataset_loaders import (
    BCICompetitionIV2aLoader,
    create_loader,
    get_loader_for_path,
    list_subjects_for_dataset,
    resolve_data_root,
)


def test_detect_bci_from_path(fixtures_dir: Path):
    path = fixtures_dir / "bci_iv2a" / "A01T.txt"
    assert BCICompetitionIV2aLoader.detect_from_path(path)
    loader = create_loader(dataset_id="bci_iv2a", n_channels=8)
    assert isinstance(loader, BCICompetitionIV2aLoader)
    assert loader.metadata.n_channels == 8


def test_unknown_path_has_no_loader(generic_sample: Path):
    with pytest.raises(ValueError):
        get_loader_for_path(generic_sample)


def test_list_and_resolve_bci_subjects(fixtures_dir: Path):
    root = fixtures_dir / "bci_iv2a"
    subjects = list_subjects_for_dataset("bci_iv2a", root)
    ids = {s.subject_id: s.runs for s in subjects}
    assert "A01" in ids
    assert set(ids["A01"]) == {"T", "E"}
    resolved = BCICompetitionIV2aLoader.resolve_subject_file(root, "A01", "T")
    assert resolved is not None
    assert resolved.name == "A01T.txt"


def test_resolve_data_root_default(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    root = resolve_data_root("bci_iv2a")
    assert root == tmp_path / "data" / "bci_iv2a"
    explicit = resolve_data_root("bci_iv2a", tmp_path / "custom")
    assert explicit == tmp_path / "custom"


def test_create_loader_unknown_id():
    with pytest.raises(ValueError):
        create_loader(dataset_id="does_not_exist")
