"""Tests for shared CSV/TXT parsing."""
from pathlib import Path

from dataset_loaders.parsing import iter_float_rows


def test_iter_float_rows_skips_comments_and_bad_values(generic_sample: Path):
    rows = list(
        iter_float_rows(generic_sample, n_channels=8, start_col=1, comment_prefix="%")
    )
    assert len(rows) == 3
    assert rows[0] == [12.3, -4.5, 0.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    assert rows[2][0] == 12.6


def test_iter_float_rows_start_col_zero(physionet_sample: Path):
    rows = list(
        iter_float_rows(physionet_sample, n_channels=8, start_col=0, comment_prefix="%")
    )
    assert len(rows) == 3
    assert rows[0][0] == 0.1


def test_iter_float_rows_short_lines_skipped(tmp_path: Path):
    p = tmp_path / "short.txt"
    p.write_text("1,2\n1,2,3,4\n", encoding="utf-8")
    rows = list(iter_float_rows(p, n_channels=4, start_col=0, comment_prefix=None))
    assert rows == [[1.0, 2.0, 3.0, 4.0]]
