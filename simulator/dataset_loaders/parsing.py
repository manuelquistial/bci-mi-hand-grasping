"""Shared CSV/TXT line parsing for EEG dataset loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_float_rows(
    file_path: str | Path,
    *,
    n_channels: int,
    start_col: int = 0,
    comment_prefix: str | None = "%",
) -> Iterator[list[float]]:
    """
    Yield rows of n_channels floats from a comma-separated text file.

    - Empty lines are skipped.
    - Lines starting with comment_prefix are skipped (if comment_prefix is set).
    - Rows with fewer than start_col + n_channels fields are skipped.
    - Rows with non-float values in the selected columns are skipped.
    """
    path = Path(file_path)
    end = start_col + n_channels
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if comment_prefix and line.startswith(comment_prefix):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < end:
                continue
            try:
                yield [float(parts[i]) for i in range(start_col, end)]
            except ValueError:
                continue
