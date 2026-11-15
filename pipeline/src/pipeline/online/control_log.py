"""Thread-safe JSON-lines event log for the live control bridge."""

from __future__ import annotations

import json
import threading
from typing import Any


class JsonlEventLog:
    def __init__(self, path: str | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        if not self.path:
            return
        line = json.dumps(row, ensure_ascii=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
