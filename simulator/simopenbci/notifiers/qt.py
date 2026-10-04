"""Qt message-box notifier (lazy PyQt5 import)."""
from __future__ import annotations


class QtNotifier:
    def __init__(self) -> None:
        self._app = None

    def _ensure_app(self):
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        self._app = app
        return app

    def info(self, message: str, *, title: str = "Info") -> None:
        from PyQt5.QtWidgets import QMessageBox

        self._ensure_app()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setInformativeText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def warning(self, message: str, *, title: str = "Warning") -> None:
        from PyQt5.QtWidgets import QMessageBox

        self._ensure_app()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(title)
        msg.setInformativeText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def error(self, message: str, *, title: str = "Error") -> None:
        from PyQt5.QtWidgets import QMessageBox

        self._ensure_app()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(title)
        msg.setInformativeText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def progress(self, message: str) -> None:
        print(message)
