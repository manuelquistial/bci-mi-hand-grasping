"""Control bridge attach and online status payload."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pipeline.models import StreamSummary
from pipeline.online.control_bridge import LiveControlBridge

if TYPE_CHECKING:
    from pipeline.application.online_session import OnlineRunManager


def maybe_start_control_bridge(manager: OnlineRunManager, summary: StreamSummary) -> None:
    with manager._lock:
        frozen = manager._frozen_model
        hub = manager._control_hub
        dataset = manager._online_dataset or "shuqfa2024"
        subject = manager._online_subject or ""
        decoder_config = manager._control_decoder_config
        if frozen is None or hub is None or manager._control_bridge is not None:
            return
        observe_hub = manager._observe_hub
    log_dir = manager.recordings_root.parent / "batch-results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"control-log-{int(time.time())}.jsonl")
    channel_labels = [str(name) for name in (summary.channel_labels or ())]
    bridge = LiveControlBridge(
        frozen,
        dataset=dataset,
        hub=hub,
        sample_rate_hz=float(summary.nominal_srate or frozen.sample_rate_hz),
        channel_count=int(summary.channel_count),
        decoder_config=decoder_config,
        subject_id=str(subject or ""),
        log_path=log_path,
        observe_hub=observe_hub,
        channel_labels=channel_labels or None,
    )
    with manager._lock:
        manager._control_bridge = bridge


def build_status_slice(manager: OnlineRunManager) -> dict[str, Any] | None:
    with manager._lock:
        proc = manager._proc
        samples = manager._online_samples
        emg_samples = manager._emg_samples
        error = manager._online_error
        source = manager._online_source
        activity = manager._windower.latest_view()
        registry = manager._control_registry
        hub = manager._control_hub
        observe_hub = manager._observe_hub
        bridge = manager._control_bridge
    if source is None:
        return None
    return {
        "source": source,
        "pid": proc.pid if proc is not None else None,
        "running": proc.poll() is None if proc is not None else True,
        "samples_seen": samples,
        "emg_samples_seen": emg_samples if source == "cyton" else 0,
        "error": error,
        "activity": activity,
        "registry": registry,
        "control_ws_url": None if hub is None else hub.url,
        "observe_ws_url": None if observe_hub is None else observe_hub.url,
        "control_samples": None if bridge is None else bridge.published_count,
        "last_control_sample": None if bridge is None else bridge.last_sample,
        "last_pipeline_stage": None if bridge is None else bridge.last_stage,
        "last_stream_frame": None if bridge is None else bridge.last_stream_frame,
    }
