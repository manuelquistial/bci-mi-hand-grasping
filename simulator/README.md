# SimOpenBCI

A simulator that sends multichannel EEG signals to **Lab Streaming Layer (LSL)** by reading data from files.

**Ports & Adapters** architecture: the core (`PlaybackService`) depends on neither Qt nor pylsl; those are adapters. Dataset formats live in `dataset_loaders`.

Environment and dependencies are managed with **[uv](https://docs.astral.sh/uv/)**.

More detail per dataset and on extending the simulator: [`docs/`](docs/README.md).

## What the code does

- **`simopenbci.PlaybackService`**: reads samples with a loader and pushes them to a `SampleSink` using drift-compensated clock timing.
- **`simopenbci.cli`** (package): `parser`, `resolve_source`, `dump`, `run_playback`; `main` is wired in `__init__.py`.
- **`dataset_loaders`**: interchangeable loaders (Shuqfa 2024, Liu 2024) with path-based detection and multi-subject support.
- **Entrypoints**: `uv run simopenbci …` or `uv run python main.py …` (default: record 02 of the first Shuqfa 2024 subject).

## Setup with uv

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11+ (the project pins `3.11` in `.python-version`).

```bash
cd simulator

# Creates .venv, installs runtime deps + the dev group (pytest) and the package in editable mode
uv sync --all-groups
```

Common commands (no need to activate the venv by hand):

```bash
uv run simopenbci --help
uv run pytest -q
uv run python main.py --no-dialog
```

If you prefer an activated shell:

```bash
source .venv/bin/activate
simopenbci --help
pytest -q
```

### Adding dependencies

```bash
uv add package-name          # runtime → pyproject.toml + uv.lock
uv add --dev package-name    # dev group
uv sync --all-groups         # re-sync after changes
```

### macOS — liblsl

`pylsl` needs the native **liblsl** library:

```bash
brew install labstreaminglayer/tap/lsl
PYTHON_SITE=$(uv run python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$PYTHON_SITE/pylsl/lib"
cp "$(brew --prefix lsl)"/lib/liblsl*.dylib "$PYTHON_SITE/pylsl/lib/"
```

## Supported datasets

### Liu et al. 2024 (grasp motor imagery, stroke)

- **Dataset**: 50 patients, 40 left/right trials, 500 Hz.
- **Usage**: `--dataset liu2024_stroke`, `--subject 1` or `--subject sub-01`.
- See [`docs/liu2024-stroke-mi.md`](docs/liu2024-stroke-mi.md).

### Shuqfa et al. 2024 (PhysioNet EEG Motor Movement/Imagery)

- **Dataset**: the cleaned PhysioNet eegmmidb: 103 subjects, 12 task runs, 64 channels, 160 Hz. One CSV per run.
- **Usage**: `--dataset shuqfa2024`, `--subject 1`, `--run 2`. Without `--run`, run `02` is sent (unilateral imagery: rest, left, right).
- See [`docs/shuqfa2024.md`](docs/shuqfa2024.md).

### Generic CSV/TXT

- Lines starting with `%` are ignored; columns 1–8 by default = 8 channels.
- Stream `BioSemi`, 8 channels, 250 Hz.

### Adding new datasets

1. Inherit from `dataset_loaders.base.BaseDatasetLoader` (`metadata`, `iter_samples`).
2. Optional: `detect_from_path()`, `default_data_root()`, `list_subjects` / `resolve_subject_file`.
3. Register in `dataset_loaders/__init__.py` → `LOADERS`.

See [`docs/multi-subject.md`](docs/multi-subject.md) for multi-subject support.

## Trial markers (LSL Markers)

If a ``<file>.events.csv`` sidecar sits next to the TXT, the simulator opens a
second LSL stream ``<eeg_name>_Markers`` and emits one marker at the instant of
each event (e.g. trial start).

Each marker is a single string with the wire format `kind:code:label`
(see `simopenbci/markers_contract.py` and [`../contracts/markers.json`](../contracts/markers.json)).

CSV format:

```csv
sample_index,code,label,kind
1200,T1,left,trial_start
1840,T0,rest,rest
```

- **Shuqfa 2024**: markers come from the annotations CSV of the same run.

```bash
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 \
  --lsl-name OpenBCI_EEG --no-dialog
```

Disable markers: `--no-markers`. Explicit path: `--events path/to/file.events.csv`.

### Session recorded by the pipeline

The pipeline writes the folder (`signal.f32`, `time.f64`, `markers.jsonl`, `session.json`). This simulator only reads it:

```bash
uv run simopenbci --dataset recorded_session --file /path/to/session \
  --lsl-name OpenBCI_EEG --no-dialog
```

The other datasets stay in their own loaders.

## Command-line usage

With no arguments, it streams record `02` of the first Shuqfa 2024 subject from `data/shuqfa2024/`:

```bash
uv run simopenbci
# headless equivalent:
uv run simopenbci --no-dialog
```

```bash
# Cleaned PhysioNet (Shuqfa 2024), unilateral imagery record
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 --no-dialog

# Specific subject / run
uv run simopenbci --dataset liu2024_stroke --subject sub-01 --no-dialog

# List subjects
uv run simopenbci --dataset shuqfa2024 --list-subjects

# Only if you want to pick a file by hand
uv run simopenbci --pick-file
```

## Requirements

- **uv** + **Python 3.11+**
- **PyQt5**: only when dialogs are used (without `--no-dialog`).
- **pylsl** (+ **liblsl** on macOS).
- **pytest**: `dev` group (`uv sync --all-groups`).

### Generic format (example)

```
%,header,optional
1, 12.3, -4.5, 0.1, 2.2, 3.3, 4.4, 5.5, 6.6
2, 12.4, -4.4, 0.2, 2.3, 3.4, 4.5, 5.6, 6.7
```

## Tests

```bash
uv sync --all-groups
uv run pytest -q
```

Tests use `RecordingSink` (no LSL). The `LslSampleSink` test is skipped when liblsl is not available.

## Connection with the pipeline

Start the simulator as an LSL source and let the pipeline (or the game via the pipeline HTTP API) consume it:

```bash
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 \
  --channels 8 --lsl-name OpenBCI_EEG --no-dialog
```

The pipeline's default LSL name is `OpenBCI_EEG`. See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) and [`../scripts/e2e_checklist.md`](../scripts/e2e_checklist.md).

## Project layout

```
simulator/
├── main.py                    # CLI entrypoint
├── simopenbci/                # Ports & Adapters core
│   ├── playback.py            # PlaybackService
│   ├── config.py              # PlaybackConfig
│   ├── ports.py               # SampleSink, Notifier, Clock
│   ├── clock.py               # RealtimeClock, InstantClock
│   ├── markers_contract.py    # kind:code:label wire contract
│   ├── sinks/                 # LslSampleSink, RecordingSink
│   ├── notifiers/             # ConsoleNotifier, QtNotifier
│   ├── gui.py                 # Qt file picker (lazy)
│   └── cli/                   # argparse + orchestration (package)
├── dataset_loaders/           # Extensible loaders
│   ├── base.py
│   ├── parsing.py             # Shared CSV/TXT parsing
│   ├── liu2024_stroke_mi.py
│   ├── shuqfa2024.py
│   ├── edf.py
│   └── generic_csv.py
├── data/                      # Local datasets (gitignored): data/<id>/
│   ├── liu2024_stroke/
│   └── shuqfa2024/
├── docs/                      # Guides (datasets, multi-subject)
│   ├── README.md
│   ├── liu2024-stroke-mi.md
│   ├── shuqfa2024.md
│   └── multi-subject.md
├── tests/                     # pytest + fixtures
├── pyproject.toml             # project + deps (uv)
├── uv.lock                    # reproducible lockfile
├── .python-version            # 3.11
└── README.md
```
