"""CLI integration tests (no pylsl required via sink_factory)."""
import json
from pathlib import Path

from simopenbci.cli import run
from simopenbci.sinks.recording import RecordingSink


def _write_signal(folder: Path, *, rows: int = 3, channels: int = 64) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    line = ",".join(str(index) for index in range(channels))
    (folder / "SUB_001_SIG_02.csv").write_text(
        "\n".join(line for _ in range(rows)) + "\n",
        encoding="utf-8",
    )


def test_list_datasets(capsys) -> None:
    code = run(["--list-datasets"])
    assert code == 0
    datasets = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in datasets] == [
        "recorded_session",
        "liu2024_stroke",
        "shuqfa2024",
    ]


def test_dump_seconds_prints_a_json_window(tmp_path: Path, capsys) -> None:
    _write_signal(tmp_path, rows=200, channels=64)
    code = run(
        [
            "--dataset",
            "shuqfa2024",
            "--data-root",
            str(tmp_path),
            "--subject",
            "S001",
            "--dump-seconds",
            "1",
            "--no-dialog",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_rate_hz"] == 160
    assert len(payload["samples"]) == 160
    assert len(payload["samples"][0]) == 64
    assert payload["channels"][0] == "Fc5"


def test_list_subjects_json(tmp_path: Path, capsys) -> None:
    _write_signal(tmp_path)
    code = run(
        [
            "--dataset",
            "shuqfa2024",
            "--list-subjects",
            "--json",
            "--data-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    subjects = json.loads(capsys.readouterr().out)
    assert subjects[0]["id"] == "S001"
    assert "02" in subjects[0]["runs"]


def test_list_subjects(tmp_path: Path):
    _write_signal(tmp_path)
    code = run(
        [
            "--dataset",
            "shuqfa2024",
            "--list-subjects",
            "--data-root",
            str(tmp_path),
        ]
    )
    assert code == 0


def test_list_subjects_defaults_to_configured_dataset():
    """--list-subjects without --dataset uses DEFAULT_DATASET."""
    code = run(["--list-subjects"])
    assert code == 0


def test_subject_resolution_and_playback(tmp_path: Path):
    _write_signal(tmp_path, channels=8)
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        s = RecordingSink()
        sinks.append(s)
        return s

    code = run(
        [
            "--dataset",
            "shuqfa2024",
            "--data-root",
            str(tmp_path),
            "--subject",
            "S001",
            "--run",
            "2",
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


def test_file_no_dialog(tmp_path: Path):
    _write_signal(tmp_path, rows=3, channels=8)
    signal = tmp_path / "SUB_001_SIG_02.csv"
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        s = RecordingSink()
        sinks.append(s)
        return s

    code = run(
        [
            "--dataset",
            "shuqfa2024",
            "--file",
            str(signal),
            "--channels",
            "8",
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
    code = run(["--no-dialog", "--dataset", "shuqfa2024"])
    assert code == 1


def test_default_uses_first_shuqfa_subject(monkeypatch, tmp_path: Path):
    """Bare --no-dialog uses DEFAULT_DATASET first subject from data/<dataset>/."""
    sample = tmp_path / "data" / "shuqfa2024"
    _write_signal(sample, rows=2)
    monkeypatch.chdir(tmp_path)

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
    assert sinks[0].metadata.n_channels == 64
