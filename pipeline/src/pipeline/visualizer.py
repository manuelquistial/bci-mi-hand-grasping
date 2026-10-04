"""Real-time multi-channel LSL signal visualizer (PyQt5 + pyqtgraph)."""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Sequence

import numpy as np

from pipeline.buffer import RingBuffer
from pipeline.models import StreamSummary
from pipeline.reader import LslStreamReader


def _channel_names(summary: StreamSummary, indices: Sequence[int]) -> list[str]:
    labels = list(summary.channel_labels)
    names: list[str] = []
    for i in indices:
        if i < len(labels) and labels[i]:
            names.append(labels[i])
        else:
            names.append(f"Ch{i + 1}")
    return names


def _parse_channels(spec: str | None, n_channels: int) -> list[int]:
    if not spec:
        return list(range(n_channels))
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    for i in indices:
        if i < 0 or i >= n_channels:
            raise ValueError(f"channel index {i} out of range [0, {n_channels - 1}]")
    if not indices:
        raise ValueError("no channels selected")
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def _curve_color(index: int, n: int) -> tuple[int, int, int]:
    hue = (index / max(1, n)) % 1.0
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _parse_marker_payload(raw: str) -> tuple[str, str]:
    """
    Parse ``kind:code:label`` (simulator convention) into (kind, display_label).
    Falls back to treating the whole string as the label.
    """
    text = (raw or "").strip()
    if not text:
        return "event", "?"
    parts = text.split(":")
    if len(parts) >= 3:
        kind, _code, label = parts[0], parts[1], ":".join(parts[2:])
        return kind or "event", label or _code or text
    if len(parts) == 2:
        return parts[0] or "event", parts[1] or text
    return "event", text


def _segment_color(kind: str, label: str) -> tuple[int, int, int]:
    """Color for a labeled epoch so rest, left, and right stay distinct."""
    key = (label or "").strip().lower()
    if key in {"left", "t1"}:
        return (72, 190, 120)
    if key in {"right", "t2"}:
        return (88, 164, 255)
    if kind == "rest" or key in {"rest", "t0"}:
        return (186, 186, 186)
    if kind == "trial_start":
        return (255, 196, 64)
    return (120, 200, 255)


def _segments_in_window(
    markers: Sequence[tuple[float, str, str]],
    *,
    last_eeg_ts: float,
    window_seconds: float,
) -> list[tuple[float, float, float, str, str, bool]]:
    """Epochs that overlap the visible window.

    Each marker opens a segment that lasts until the next marker. The latest
    segment stays open up to the newest EEG sample (plot x = 0).

    Returns ``(fill_x0, fill_x1, boundary_x, kind, label, boundary_visible)``.
    Coordinates are seconds relative to the newest sample, clipped to the window.
    """
    ordered = sorted(markers, key=lambda item: item[0])
    left = -float(window_seconds)
    right = 0.0
    segments: list[tuple[float, float, float, str, str, bool]] = []
    for index, (timestamp, kind, label) in enumerate(ordered):
        boundary = float(timestamp) - last_eeg_ts
        if index + 1 < len(ordered):
            end = float(ordered[index + 1][0]) - last_eeg_ts
        else:
            end = right
        if end < left or boundary > right:
            continue
        fill_x0 = max(boundary, left)
        fill_x1 = min(end, right)
        if fill_x1 <= fill_x0:
            continue
        segments.append(
            (fill_x0, fill_x1, boundary, kind, label, left <= boundary <= right)
        )
    return segments


def _open_marker_reader(
    *,
    eeg_name: str,
    markers_name: str | None,
    wait_seconds: float,
) -> LslStreamReader | None:
    candidates = []
    if markers_name:
        candidates.append(markers_name)
    candidates.append(f"{eeg_name}_Markers")
    last_error: Exception | None = None
    for name in candidates:
        try:
            return LslStreamReader.open(name=name, wait_seconds=min(wait_seconds, 2.0))
        except TimeoutError as exc:
            last_error = exc
            continue
    # Optional: discover any Markers stream
    try:
        return LslStreamReader.open(stream_type="Markers", wait_seconds=min(wait_seconds, 1.5))
    except TimeoutError:
        if last_error:
            print(f"No Markers stream found (optional). Last error: {last_error}")
        else:
            print("No Markers stream found (optional).")
        return None


class LslVisualizerApp:
    """Qt application that polls an LSL inlet and scrolls waveforms."""

    def __init__(
        self,
        reader: LslStreamReader,
        *,
        marker_reader: LslStreamReader | None = None,
        window_seconds: float = 5.0,
        refresh_hz: float = 30.0,
        channel_indices: Sequence[int] | None = None,
        scale: float | None = None,
    ) -> None:
        from PyQt5 import QtCore, QtWidgets
        import pyqtgraph as pg

        self._reader = reader
        self._marker_reader = marker_reader
        self._summary = reader.summary
        self._n_total = self._summary.channel_count
        self._channel_indices = list(channel_indices or range(self._n_total))
        self._n_plot = len(self._channel_indices)
        self._names = _channel_names(self._summary, self._channel_indices)

        srate = self._summary.nominal_srate
        if srate <= 0:
            srate = 250.0
        self._srate = float(srate)
        self._window_seconds = max(0.5, float(window_seconds))
        capacity = max(32, int(self._srate * self._window_seconds))
        self._buffer = RingBuffer(self._n_plot, capacity)
        self._scale = scale
        self._samples_seen = 0
        self._t0 = time.monotonic()
        self._last_eeg_ts: float | None = None
        self._markers: deque[tuple[float, str, str]] = deque(maxlen=500)
        self._marker_artists: list = []
        self._current_segment = "—"

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True, background="#121212", foreground="#e6e6e6")

        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(
            f"pipeline — LSL · {self._summary.name} · "
            f"{self._n_plot}/{self._n_total} ch · {self._srate:g} Hz"
        )
        self._win.resize(1100, max(420, 64 * min(self._n_plot, 18)))

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._status = QtWidgets.QLabel("Connecting…")
        self._status.setStyleSheet("color: #c8c8c8; font-size: 12px;")
        layout.addWidget(self._status)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)
        self._win.setCentralWidget(central)

        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Amplitude (offset per channel)")
        self._plot.setXRange(-self._window_seconds, 0.0)
        self._plot.addLegend(offset=(10, 10))
        self._pg = pg

        self._curves = []
        for i, name in enumerate(self._names):
            color = _curve_color(i, self._n_plot)
            pen = pg.mkPen(color=color, width=1.2)
            self._curves.append(self._plot.plot(pen=pen, name=name))

        self._timer = QtCore.QTimer(self._win)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(max(10, int(1000.0 / max(1.0, refresh_hz))))

        self._win.closeEvent = self._wrap_close(self._win.closeEvent)  # type: ignore[method-assign]

    def _wrap_close(self, original):  # noqa: ANN001, ANN202
        def handler(event) -> None:  # noqa: ANN001
            self._timer.stop()
            self._reader.close()
            if self._marker_reader is not None:
                self._marker_reader.close()
            original(event)

        return handler

    def _pull_markers(self) -> None:
        if self._marker_reader is None:
            return
        chunk = self._marker_reader.pull_chunk(timeout=0.0, max_samples=64)
        for sample, ts in zip(chunk.samples, chunk.timestamps):
            raw = sample[0] if sample else ""
            if not isinstance(raw, str):
                raw = str(raw)
            kind, label = _parse_marker_payload(raw)
            self._markers.append((float(ts), kind, label))
            self._current_segment = label

    def _redraw_markers(self) -> None:
        for artist in self._marker_artists:
            self._plot.removeItem(artist)
        self._marker_artists.clear()

        if self._last_eeg_ts is None:
            return

        y_top = self._plot.viewRange()[1][1]
        for fill_x0, fill_x1, boundary, kind, label, boundary_visible in _segments_in_window(
            self._markers,
            last_eeg_ts=self._last_eeg_ts,
            window_seconds=self._window_seconds,
        ):
            color = _segment_color(kind, label)
            region = self._pg.LinearRegionItem(
                values=(fill_x0, fill_x1),
                orientation="vertical",
                movable=False,
                brush=self._pg.mkBrush(*color, 48),
                pen=self._pg.mkPen(None),
            )
            region.setZValue(-20)
            self._plot.addItem(region)
            self._marker_artists.append(region)
            if not boundary_visible:
                continue
            line = self._pg.InfiniteLine(
                pos=boundary,
                angle=90,
                pen=self._pg.mkPen(color=color, width=2.0),
            )
            line.setZValue(10)
            self._plot.addItem(line)
            self._marker_artists.append(line)
            text = self._pg.TextItem(
                text=label,
                color=color,
                anchor=(0.0, 1.0),
                fill=(18, 18, 18, 200),
            )
            text.setPos(boundary, y_top)
            self._plot.addItem(text)
            self._marker_artists.append(text)

    def _on_tick(self) -> None:
        chunk = self._reader.pull_chunk(timeout=0.0, max_samples=2048)
        if chunk.samples:
            arr = np.asarray(chunk.samples, dtype=np.float64)
            selected = arr[:, self._channel_indices]
            self._buffer.append(selected)
            self._samples_seen += selected.shape[0]
            if chunk.timestamps:
                self._last_eeg_ts = float(chunk.timestamps[-1])

        self._pull_markers()

        data = self._buffer.latest()
        n = data.shape[1]
        if n == 0:
            self._status.setText("Waiting for samples…")
            return

        if self._scale is None:
            amp = float(np.percentile(np.abs(data), 95)) if data.size else 1.0
            spacing = max(amp * 2.5, 1e-6)
        else:
            spacing = float(self._scale)

        t = np.linspace(-n / self._srate, 0.0, n, endpoint=False)
        for i, curve in enumerate(self._curves):
            y = data[i] + (self._n_plot - 1 - i) * spacing
            curve.setData(t, y)

        self._plot.setYRange(-0.5 * spacing, (self._n_plot - 0.5) * spacing, padding=0.02)
        self._redraw_markers()

        elapsed = max(1e-6, time.monotonic() - self._t0)
        rate = self._samples_seen / elapsed
        markers_info = (
            f"markers=on · segment={self._current_segment}"
            if self._marker_reader is not None
            else "markers=off"
        )
        self._status.setText(
            f"{self._summary.name}  ·  type={self._summary.stream_type}  ·  "
            f"{self._n_plot}/{self._n_total} ch @ {self._srate:g} Hz  ·  "
            f"buffer {n}/{self._buffer.capacity}  ·  ~{rate:.0f} samp/s  ·  "
            f"window {self._window_seconds:g}s  ·  {markers_info}"
        )

    def run(self) -> int:
        self._win.show()
        return int(self._app.exec_())


def run_visualizer(
    *,
    name: str | None = None,
    stream_type: str | None = "EEG",
    wait_seconds: float = 5.0,
    window_seconds: float = 5.0,
    refresh_hz: float = 30.0,
    channels: str | None = None,
    scale: float | None = None,
    markers_name: str | None = None,
    no_markers: bool = False,
) -> int:
    """Connect to LSL and open the live viewer window."""
    print(
        f"Looking for LSL stream "
        f"(name={name!r}, type={stream_type!r}, wait={wait_seconds}s)…",
        flush=True,
    )
    reader = LslStreamReader.open(
        name=name,
        stream_type=None if name else stream_type,
        wait_seconds=wait_seconds,
    )
    print(f"Connected: {reader.summary.display_name}", flush=True)

    marker_reader = None
    if not no_markers:
        print("Looking for Markers stream…", flush=True)
        marker_reader = _open_marker_reader(
            eeg_name=reader.summary.name,
            markers_name=markers_name,
            wait_seconds=wait_seconds,
        )
        if marker_reader is not None:
            print(f"Markers: {marker_reader.summary.display_name}", flush=True)

    indices = _parse_channels(channels, reader.summary.channel_count)
    app = LslVisualizerApp(
        reader,
        marker_reader=marker_reader,
        window_seconds=window_seconds,
        refresh_hz=refresh_hz,
        channel_indices=indices,
        scale=scale,
    )
    return app.run()
