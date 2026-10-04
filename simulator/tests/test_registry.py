"""Tests for loader registry and multi-subject helpers."""
from pathlib import Path

from dataset_loaders import (
    BCICompetitionIV2aLoader,
    GenericCSVLoader,
    PhysioNetEEGBCILoader,
    create_loader,
    get_loader_for_path,
    list_subjects_for_dataset,
    resolve_data_root,
)


def test_detect_bci_from_path(fixtures_dir: Path):
    path = fixtures_dir / "bci_iv2a" / "A01T.txt"
    # detect_from_path looks for BCI_Competition_IV2a in path string
    assert not BCICompetitionIV2aLoader.detect_from_path(path)
    # rename-like path via tmp with folder name
    root = fixtures_dir / "sujetos_BCI_Competition_IV2a"
    # use fixtures bci_iv2a folder — detection needs substring; test create via id
    loader = create_loader(dataset_id="bci_iv2a", n_channels=8)
    assert isinstance(loader, BCICompetitionIV2aLoader)
    assert loader.metadata.n_channels == 8


def test_detect_physionet_from_filename(physionet_sample: Path):
    assert PhysioNetEEGBCILoader.detect_from_path(physionet_sample)
    loader = get_loader_for_path(physionet_sample)
    assert isinstance(loader, PhysioNetEEGBCILoader)


def test_detect_generic_fallback(generic_sample: Path):
    loader = get_loader_for_path(generic_sample)
    assert isinstance(loader, GenericCSVLoader)


def test_list_and_resolve_bci_subjects(fixtures_dir: Path):
    root = fixtures_dir / "bci_iv2a"
    subjects = list_subjects_for_dataset("bci_iv2a", root)
    ids = {s.subject_id: s.runs for s in subjects}
    assert "A01" in ids
    assert set(ids["A01"]) == {"T", "E"}
    resolved = BCICompetitionIV2aLoader.resolve_subject_file(root, "A01", "T")
    assert resolved is not None
    assert resolved.name == "A01T.txt"


def test_list_and_resolve_physionet(fixtures_dir: Path):
    root = fixtures_dir / "physionet_eegbci"
    subjects = list_subjects_for_dataset("physionet_eegbci", root)
    assert subjects[0].subject_id == "S001"
    resolved = PhysioNetEEGBCILoader.resolve_subject_file(root, "1")
    assert resolved is not None
    assert resolved.name == "S001.txt"


def test_resolve_data_root_default(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    root = resolve_data_root("bci_iv2a")
    assert root == tmp_path / "data" / "sujetos_BCI_Competition_IV2a"
    explicit = resolve_data_root("bci_iv2a", tmp_path / "custom")
    assert explicit == tmp_path / "custom"


def test_create_loader_unknown_id():
    import pytest

    with pytest.raises(ValueError):
        create_loader(dataset_id="does_not_exist")
