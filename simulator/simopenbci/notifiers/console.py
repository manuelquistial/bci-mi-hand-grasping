"""Console notifier (headless / --no-dialog)."""
from __future__ import annotations


class ConsoleNotifier:
    def info(self, message: str, *, title: str = "Info") -> None:
        print(message)

    def warning(self, message: str, *, title: str = "Warning") -> None:
        print(f"Warning: {message}")

    def error(self, message: str, *, title: str = "Error") -> None:
        print(f"Error: {message}")

    def progress(self, message: str) -> None:
        print(message)
