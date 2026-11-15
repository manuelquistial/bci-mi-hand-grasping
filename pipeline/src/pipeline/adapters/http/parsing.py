"""Request-body parsing helpers for the HTTP adapter."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def number(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
