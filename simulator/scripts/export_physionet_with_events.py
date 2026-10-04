"""Export PhysioNet EEGBCI runs to simulator TXT + events sidecar."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PHYSIONET_LABELS = {
    "T0": ("rest", "rest"),
    "T1": ("left", "trial_start"),
    "T2": ("right", "trial_start"),
}


def export_subject(
    *,
    edf_dir: Path,
    subject: int,
    runs: list[int],
    out_dir: Path,
    n_channels: int = 8,
) -> tuple[Path, Path]:
    import mne
    import numpy as np

    mne.set_log_level("ERROR")
    sid = f"S{subject:03d}"
    chunks: list = []
    events_rows: list[tuple[int, str, str, str]] = []
    offset = 0

    for run in runs:
        fpath = edf_dir / sid / f"{sid}R{run:02d}.edf"
        if not fpath.is_file():
            # Some installs flatten files under subject folder without nested sid
            fpath = edf_dir / f"{sid}R{run:02d}.edf"
        if not fpath.is_file():
            raise FileNotFoundError(fpath)

        raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        raw.pick(raw.ch_names[:n_channels])
        data = (raw.get_data() * 1e6).T  # µV
        n_samples = data.shape[0]
        chunks.append(data)

        try:
            ann = raw.annotations
        except Exception:
            ann = None
        if ann is not None:
            sfreq = float(raw.info["sfreq"])
            for onset, desc in zip(ann.onset, ann.description):
                code = str(desc).strip()
                label, kind = PHYSIONET_LABELS.get(code, (code, "event"))
                sample_index = offset + int(round(float(onset) * sfreq))
                sample_index = min(max(sample_index, 0), offset + n_samples - 1)
                events_rows.append((sample_index, code, label, kind))
        offset += n_samples

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{sid}.txt"
    events_path = out_dir / f"{sid}.events.csv"
    arr = np.vstack(chunks)
    np.savetxt(txt_path, arr, delimiter=",", fmt="%.6f")

    with events_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "code", "label", "kind"])
        writer.writerows(events_rows)

    return txt_path, events_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edf-dir",
        type=Path,
        required=True,
        help="Folder containing S001/… or flat S001R04.edf files",
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="PhysioNet run numbers (default: 4 8 12 MI left/right)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/physionet_eegbci"),
    )
    args = parser.parse_args(argv)
    txt, events = export_subject(
        edf_dir=args.edf_dir,
        subject=args.subject,
        runs=args.runs,
        out_dir=args.out_dir,
    )
    print(f"Wrote {txt}")
    print(f"Wrote {events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
