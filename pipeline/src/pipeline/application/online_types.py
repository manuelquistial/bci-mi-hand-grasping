"""Request types for online session start."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.config import CONTROL_WS_PORT


@dataclass(frozen=True)
class OnlineStartRequest:
    source: str
    lsl_name: str
    dataset: str | None = None
    session_dir: str | None = None
    file: str | None = None
    subject: str | None = None
    data_root: str | None = None
    run: str | None = None
    max_samples: int | None = None
    wait_seconds: float = 5.0
    registry: str | None = None
    control_ws_port: int = CONTROL_WS_PORT
    observe_ws_port: int | None = None
    control_strategy: str = "baseline_proba"
    control_window_k: int = 3
    control_proportional_scale: float = 1.0
