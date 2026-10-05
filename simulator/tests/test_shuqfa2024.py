"""Shuqfa 2024 curated PhysioNet: one CSV per task run."""

from pathlib import Path

from dataset_loaders.shuqfa2024 import Shuqfa2024Loader
from simopenbci.cli import run
from simopenbci.sinks.recording import RecordingSink


def _write_pair(folder: Path, subject: str, run: str) -> None:
    signal = folder / f"SUB_{subject}_SIG_{run}.csv"
    rows = []
    for index in range(6):
        rows.append(",".join(str(index + channel / 100) for channel in range(64)))
    signal.write_text("\n".join(rows) + "\n", encoding="utf-8")
    annotations = folder / f"SUB_{subject}_ANN_{run}.csv"
    annotations.write_text(
        "4,4.0,2,1,2\n"
        "5,4.0,2,3,4\n"
        "6,4.1,2,5,6\n",
        encoding="utf-8",
    )


def test_annotations_set_marker_index_and_labels(tmp_path: Path) -> None:
    _write_pair(tmp_path, "001", "02")
    signal = tmp_path / "SUB_001_SIG_02.csv"
    loader = Shuqfa2024Loader()
    rows = list(loader.iter_samples(signal))
    assert len(rows) == 6
    assert len(rows[0]) == 64
    assert loader.metadata.sampling_rate == 160.0
    assert loader.metadata.channel_names[0] == "Fc5"
    events = loader.load_events(signal)
    assert [(event.sample_index, event.code, event.label, event.kind) for event in events] == [
        (0, "4", "rest", "rest"),
        (2, "5", "left", "trial_start"),
        (4, "6", "right", "trial_start"),
    ]


def test_list_and_resolve_runs(tmp_path: Path) -> None:
    _write_pair(tmp_path, "001", "01")
    _write_pair(tmp_path, "001", "02")
    _write_pair(tmp_path, "001", "10")
    subjects = Shuqfa2024Loader.list_subjects(tmp_path)
    assert subjects[0].subject_id == "S001"
    assert subjects[0].runs == ["01", "02", "10"]
    default = Shuqfa2024Loader.resolve_subject_file(tmp_path, "1")
    imagery = Shuqfa2024Loader.resolve_subject_file(tmp_path, "S001", "10")
    assert default is not None and default.name == "SUB_001_SIG_02.csv"
    assert imagery is not None and imagery.name == "SUB_001_SIG_10.csv"


def test_cli_plays_one_run(tmp_path: Path) -> None:
    _write_pair(tmp_path, "004", "06")
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        sink = RecordingSink()
        sinks.append(sink)
        return sink

    code = run(
        [
            "--dataset",
            "shuqfa2024",
            "--data-root",
            str(tmp_path),
            "--subject",
            "S004",
            "--run",
            "6",
            "--channels",
            "8",
            "--max-samples",
            "2",
            "--no-dialog",
            "--no-markers",
        ],
        sink_factory=factory,
    )
    assert code == 0
    assert len(sinks[0].samples) == 2
    assert len(sinks[0].samples[0]) == 8
    assert sinks[0].metadata is not None
    assert sinks[0].metadata.sampling_rate == 160.0
