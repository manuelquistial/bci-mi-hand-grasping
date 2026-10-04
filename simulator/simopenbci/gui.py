"""Lazy Qt file picker."""
from __future__ import annotations

from pathlib import Path


class QtFilePicker:
    def pick_file(self) -> str | None:
        from PyQt5.QtWidgets import QApplication, QFileDialog

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select an EEG file",
            str(Path.cwd()),
            "Text Files (*.txt *.csv);;All Files (*)",
        )
        return file_path or None
