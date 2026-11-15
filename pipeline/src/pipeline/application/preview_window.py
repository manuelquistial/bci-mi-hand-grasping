"""One filtered preview window for one participant: raw, filtered, spatial."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.adapters.simulator.catalog import query_simulator_window
from pipeline.params import ModelParams
from pipeline.preprocess import (
    PREVIEW_SECONDS,
    FilterSpec,
    apply_notch_then_bandpass,
    decimate_time,
    traces,
)
from pipeline.spatial import apply_spatial

WindowLoader = Callable[..., dict[str, Any]]


class PreviewWindows:
    """Load a short window and run the notch, bandpass, and spatial steps."""

    def __init__(self, *, simulator_project: Path, window_loader: WindowLoader | None) -> None:
        self._simulator_project = simulator_project
        self._window_loader = window_loader

    def prepare(
        self,
        *,
        dataset: str,
        subject: str | None,
        session_dir: str | None,
        notch_hz: float,
        band_low_hz: float,
        band_high_hz: float,
        order: int,
        spatial: str,
        params: ModelParams | None = None,
    ) -> dict[str, Any]:
        loader = self._window_loader or (
            lambda **kwargs: query_simulator_window(self._simulator_project, **kwargs)
        )
        window = loader(
            dataset=dataset,
            subject=subject,
            session_dir=session_dir,
            seconds=PREVIEW_SECONDS,
        )
        samples = np.asarray(window["samples"], dtype=np.float64)
        rate = float(window["sample_rate_hz"])
        tuned = params or ModelParams()
        spec = FilterSpec(
            notch_hz=float(notch_hz),
            notch_q=float(tuned.notch_q),
            band_low_hz=float(band_low_hz),
            band_high_hz=float(band_high_hz),
            order=int(order),
        )
        filtered = apply_notch_then_bandpass(samples, rate, spec)
        raw_view = decimate_time(samples)
        channels = [str(name) for name in window.get("channels") or []]
        while len(channels) < int(raw_view.shape[1]):
            channels.append(f"Ch{len(channels) + 1}")
        channels = channels[: int(raw_view.shape[1])]
        spatial_name = (spatial or "none").strip().lower()
        spatial_labels: list[str] = []
        spatial_view: list[list[float]] = []
        current = filtered
        current_channels = channels
        if spatial_name != "none":
            current, spatial_labels = apply_spatial(filtered, channels, spatial_name, tuned)
            current_channels = spatial_labels
            spatial_view = traces(decimate_time(current))
        return {
            "rate": rate,
            "spec": spec,
            "seconds": round(samples.shape[0] / rate, 3) if rate else 0,
            "channels": channels,
            "raw_view": raw_view,
            "filtered_view": decimate_time(filtered),
            "spatial_name": spatial_name,
            "spatial_labels": spatial_labels,
            "spatial_view": spatial_view,
            "current": current,
            "current_channels": current_channels,
        }
