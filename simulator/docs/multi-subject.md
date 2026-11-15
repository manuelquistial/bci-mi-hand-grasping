# Datasets with multiple subjects

For a dataset to work with **every subject** from the command line (`--list-subjects`, `--subject`, `--run`), the loader must implement two class methods from the **base** class:

## 1. `list_subjects(data_root)` → `list[SubjectRunInfo]`

- **Input**: `data_root`, the dataset root folder (e.g. `data/shuqfa2024`).
- **Output**: a list of `SubjectRunInfo(subject_id="S001", runs=["01", "02"])` covering every subject and the runs/sessions available for each.
- **Use**: called by `--list-subjects` to print the table of subjects and runs.

## 2. `resolve_subject_file(data_root, subject_id, run=None)` → `Path | None`

- **Input**: the same `data_root`, a `subject_id` (e.g. `"S001"`), and optionally a `run` (e.g. `"02"`).
- **Output**: the path of the file to stream for that subject and run, or `None` if it does not exist.
- **Use**: with `--subject S001 --run 02` the simulator uses this path instead of `--file`.

## 3. `default_data_root()` → `Path | None` (optional)

- Returns `data/<id>/` under the current working directory (e.g. `data/shuqfa2024`).
- If implemented, `--list-subjects` and `--subject` do not require `--data-root`.

## Naming convention

- Each dataset chooses its own scheme. Shuqfa 2024 uses `SUB_001_SIG_02.csv`.
- You only need to implement `list_subjects` and `resolve_subject_file` consistently with those file names.

## Example: adding an "OpenBMI" dataset with subjects

1. Create `openbmi_loader.py` with a class that inherits from `BaseDatasetLoader`.
2. Implement `iter_samples` and `metadata`, and, where applicable, `detect_from_path` / `default_data_root`.
3. For multi-subject support, implement:
   - `list_subjects(data_root)`: scan `data_root` and return the subjects and their runs.
   - `resolve_subject_file(data_root, subject_id, run)`: return the `Path` of the matching file.
4. Register the loader in `__init__.py` under `LOADERS`.

Then `uv run simopenbci --dataset openbmi --list-subjects` and `--subject X --run Y` work **without changing** `simopenbci/cli/`.
