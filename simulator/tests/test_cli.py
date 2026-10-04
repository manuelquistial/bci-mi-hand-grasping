"""CLI integration tests (no pylsl required via sink_factory)."""
from pathlib import Path

from simopenbci.cli import run
from simopenbci.sinks.recording import RecordingSink


def test_list_subjects(fixtures_dir: Path):
    code = run(
        [
            "--dataset",
            "bci_iv2a",
            "--list-subjects",
            "--data-root",
            str(fixtures_dir / "bci_iv2a"),
        ]
    )
    assert code == 0


def test_list_subjects_defaults_to_bci():
    """--list-subjects without --dataset uses DEFAULT_DATASET."""
    code = run(["--list-subjects"])
    assert code == 0


def test_subject_resolution_and_playback(fixtures_dir: Path):
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        s = RecordingSink()
        sinks.append(s)
        return s

    code = run(
        [
            "--dataset",
            "bci_iv2a",
            "--data-root",
            str(fixtures_dir / "bci_iv2a"),
            "--subject",
            "A01",
            "--run",
            "T",
            "--channels",
            "8",
            "--max-samples",
            "2",
            "--no-dialog",
        ],
        sink_factory=factory,
    )
    assert code == 0
    assert len(sinks) == 1
    assert len(sinks[0].samples) == 2
    assert len(sinks[0].samples[0]) == 8


def test_file_no_dialog(generic_sample: Path):
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        s = RecordingSink()
        sinks.append(s)
        return s

    code = run(
        [
            "--dataset",
            "generic_csv",
            "--file",
            str(generic_sample),
            "--max-samples",
            "10",
            "--no-dialog",
        ],
        sink_factory=factory,
    )
    assert code == 0
    assert len(sinks[0].samples) == 3


def test_no_dialog_without_file_or_dataset_data_fails(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = run(["--no-dialog", "--dataset", "generic_csv"])
    assert code == 1


def test_default_uses_first_bci_subject(monkeypatch, fixtures_dir: Path):
    """Bare --no-dialog uses DEFAULT_DATASET first subject from sample_data/data."""
    # Point cwd at a tree with sample_data/bci_iv2a like the repo
    root = fixtures_dir.parent.parent  # tests/ -> project; fixtures are under tests/
    # Use fixtures dir as data-root via monkeypatch of default by chdir into a temp
    # with sample_data layout
    import tempfile
    from pathlib import Path as P
    import shutil

    work = P(tempfile.mkdtemp())
    sample = work / "sample_data" / "bci_iv2a"
    sample.mkdir(parents=True)
    shutil.copy(fixtures_dir / "bci_iv2a" / "A01T.txt", sample / "A01T.txt")
    monkeypatch.chdir(work)

    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        s = RecordingSink()
        sinks.append(s)
        return s

    code = run(["--no-dialog", "--max-samples", "2"], sink_factory=factory)
    assert code == 0
    assert len(sinks) == 1
    assert len(sinks[0].samples) == 2
    assert sinks[0].metadata is not None
    assert sinks[0].metadata.n_channels == 22
