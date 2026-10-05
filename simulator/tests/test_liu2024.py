"""Liu 2024 stroke grasp-imagery loader."""

import struct
from pathlib import Path

from dataset_loaders.edf import read_header
from dataset_loaders.liu2024_stroke_mi import FILE_CHANNEL_LABELS, Liu2024StrokeMILoader
from simopenbci.cli import run
from simopenbci.sinks.recording import RecordingSink


def _ascii(text: str, width: int) -> bytes:
    raw = text.encode("ascii")[:width]
    return raw.ljust(width)


def _write_edf(path: Path, samples: list[list[int]]) -> None:
    """Write a 33-channel EDF. ``samples[channel]`` is the full int16 series."""
    labels = list(FILE_CHANNEL_LABELS)
    n_signals = len(labels)
    spr = 10
    n_records = len(samples[0]) // spr
    header_bytes = 256 + n_signals * 256
    main = b"".join(
        [
            _ascii("0", 8),
            _ascii("X", 80),
            _ascii("Startdate 01-JAN-2024", 80),
            _ascii("01.01.24", 8),
            _ascii("00.00.00", 8),
            _ascii(str(header_bytes), 8),
            _ascii("", 44),
            _ascii(str(n_records), 8),
            _ascii("1", 8),
            _ascii(str(n_signals), 4),
        ]
    )
    assert len(main) == 256

    def signal_field(values: list[str], width: int) -> bytes:
        return b"".join(_ascii(value, width) for value in values)

    physical_min: list[str] = []
    physical_max: list[str] = []
    for label in labels:
        if label == "CPz":
            physical_min.append("0")
            physical_max.append("0")
        elif label == "":
            physical_min.append("0")
            physical_max.append("3")
        else:
            physical_min.append("-100")
            physical_max.append("100")
    channels = b"".join(
        [
            signal_field(labels, 16),
            signal_field(["AgAgCl"] * n_signals, 80),
            signal_field(["uV"] * n_signals, 8),
            signal_field(physical_min, 8),
            signal_field(physical_max, 8),
            signal_field(["-32768"] * n_signals, 8),
            signal_field(["32767"] * n_signals, 8),
            signal_field([""] * n_signals, 80),
            signal_field([str(spr)] * n_signals, 8),
            signal_field([""] * n_signals, 32),
        ]
    )
    assert len(channels) == n_signals * 256
    body = bytearray()
    for record in range(n_records):
        for channel in range(n_signals):
            chunk = samples[channel][record * spr : (record + 1) * spr]
            body.extend(struct.pack("<" + "h" * len(chunk), *chunk))
    path.write_bytes(main + channels + body)


def _marker_digital(code: int) -> int:
    if code == 0:
        return -32768
    if code == 3:
        return 32767
    return int(round(code * 65535 / 3 - 32768))


def _synthetic_edf(path: Path) -> None:
    n_signals = len(FILE_CHANNEL_LABELS)
    total = 20
    samples = [[-32768 + channel] * total for channel in range(n_signals)]
    marker = [_marker_digital(0)] * total
    for index, code in ((0, 1), (5, 2), (8, 3), (10, 1), (15, 2), (18, 3)):
        marker[index] = _marker_digital(code)
    samples[-1] = marker
    _write_edf(path, samples)


def _physical(channel: int) -> float:
    return channel * 200 / 65535 - 100


def test_loader_skips_reference_and_reads_markers(tmp_path: Path) -> None:
    path = tmp_path / "sub-01_task-motor-imagery_eeg.edf"
    _synthetic_edf(path)
    header = read_header(path)
    assert header.sampling_rate == 10
    assert len(header.signals) == 33

    loader = Liu2024StrokeMILoader()
    rows = list(loader.iter_samples(path))
    assert len(rows) == 20
    assert len(rows[0]) == 29
    assert loader.metadata.channel_names is not None
    assert "CPz" not in loader.metadata.channel_names
    assert "HEOL" not in loader.metadata.channel_names
    assert "HEOR" not in loader.metadata.channel_names
    assert loader.metadata.channel_names[0] == "FP1"
    assert loader.metadata.channel_names[16] == "T4"
    assert loader.metadata.channel_names[17] == "CP3"
    assert loader.metadata.channel_names[-3:] == ["Oz", "O1", "O2"]
    assert rows[0][0] == _physical(0)
    assert rows[0][17] == _physical(18)

    events = loader.load_events(path)
    assert [(event.sample_index, event.kind, event.label) for event in events] == [
        (0, "instruction", "left"),
        (5, "trial_start", "left"),
        (8, "rest", "rest"),
        (10, "instruction", "right"),
        (15, "trial_start", "right"),
        (18, "rest", "rest"),
    ]


def test_list_and_resolve_subjects(tmp_path: Path) -> None:
    first = tmp_path / "edffile" / "sub-01" / "eeg"
    second = tmp_path / "edffile" / "sub-02" / "eeg"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _synthetic_edf(first / "sub-01_task-motor-imagery_eeg.edf")
    _synthetic_edf(second / "sub-02_task-motor-imagery_eeg.edf")

    subjects = Liu2024StrokeMILoader.list_subjects(tmp_path)
    assert [item.subject_id for item in subjects] == ["sub-01", "sub-02"]
    assert subjects[0].runs == ["imagery"]
    resolved = Liu2024StrokeMILoader.resolve_subject_file(tmp_path, "1")
    assert resolved is not None
    assert resolved.name == "sub-01_task-motor-imagery_eeg.edf"
    assert Liu2024StrokeMILoader.resolve_subject_file(tmp_path, "sub-02", "imagery") == (
        second / "sub-02_task-motor-imagery_eeg.edf"
    )


def test_cli_plays_subject(tmp_path: Path) -> None:
    folder = tmp_path / "edffile" / "sub-03" / "eeg"
    folder.mkdir(parents=True)
    _synthetic_edf(folder / "sub-03_task-motor-imagery_eeg.edf")
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        sink = RecordingSink()
        sinks.append(sink)
        return sink

    code = run(
        [
            "--dataset",
            "liu2024_stroke",
            "--data-root",
            str(tmp_path),
            "--subject",
            "sub-03",
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
    assert sinks[0].metadata.sampling_rate == 10
