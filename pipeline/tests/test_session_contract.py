"""The folder written by the pipeline is readable by the simulator loader."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from pipeline.session import SessionWriter

SIMULATOR = Path(__file__).resolve().parents[2] / "simulator"


def test_simulator_loader_reads_a_pipeline_session(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / "sess1"
    writer = SessionWriter(
        root,
        session_id="sess1",
        channel_labels=tuple(f"Ch{index + 1}" for index in range(8)),
        sample_rate_hz=250.0,
        stream_name="OpenBCI_EEG",
    )
    samples = np.array([[float(i) for i in range(8)], [float(i) for i in range(8, 16)]])
    writer.append_samples(samples, [10.0, 10.004])
    writer.append_marker(10.006, "trial_start", "T2", "right")
    writer.close()

    script = """
from dataset_loaders.recorded_session import RecordedSessionLoader
import sys
loader = RecordedSessionLoader()
samples = list(loader.iter_samples(sys.argv[1]))
events = loader.load_events(sys.argv[1])
print(len(samples), len(samples[0]), round(samples[1][0], 3), events[0].label, events[0].sample_index, events[0].kind)
"""
    completed = subprocess.run(
        ["uv", "run", "--project", str(SIMULATOR), "python", "-c", script, str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "2 8 8.0 right 1 trial_start"
