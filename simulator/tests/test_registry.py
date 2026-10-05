"""Tests for loader registry and multi-subject helpers."""
from pathlib import Path

import pytest

from dataset_loaders import (
    Shuqfa2024Loader,
    create_loader,
    get_loader_for_path,
    resolve_data_root,
)


def test_detect_shuqfa_from_path(tmp_path: Path):
    path = tmp_path / "SUB_001_SIG_02.csv"
    path.write_text("0\n", encoding="utf-8")
    assert Shuqfa2024Loader.detect_from_path(path)
    loader = create_loader(dataset_id="shuqfa2024", n_channels=8)
    assert isinstance(loader, Shuqfa2024Loader)
    assert loader.metadata.n_channels == 8


def test_unknown_path_has_no_loader(generic_sample: Path):
    with pytest.raises(ValueError):
        get_loader_for_path(generic_sample)


def test_resolve_data_root_default(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    root = resolve_data_root("shuqfa2024")
    assert root == tmp_path / "data" / "shuqfa2024"
    explicit = resolve_data_root("shuqfa2024", tmp_path / "custom")
    assert explicit == tmp_path / "custom"


def test_create_loader_unknown_id():
    with pytest.raises(ValueError):
        create_loader(dataset_id="does_not_exist")
