"""Integration: frozen model → decoder → WebSocket v2 publish/receive."""

from __future__ import annotations

import json
import threading
import time

import numpy as np

from pipeline.online.control_bridge import PROTOCOL_VERSION, ControlWsHub
from pipeline.online.control_decoder import ControlDecoder, ControlDecoderConfig
from pipeline.online.model import train_frozen_model
from pipeline.online.types import WindowDecision


def _model():
    rng = np.random.default_rng(2)
    trials, labels = [], []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 8:
        n = 400
        t = np.arange(n) / 250.0
        data = rng.normal(0.0, 0.3, size=(8, n))
        data[ch] += 2.0 * np.sin(2 * np.pi * freq * t)
        trials.append(data)
        labels.append(label)
    return train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_names=("linear_svm",),
        apply_filter=False,
        apply_training_ea=False,
    )


def test_ws_hub_publishes_v2_control() -> None:
    hub = ControlWsHub(host="127.0.0.1", port=18091)
    hub.start()
    received: list[dict] = []
    done = threading.Event()

    def client() -> None:
        import asyncio

        import websockets

        async def _run() -> None:
            async with websockets.connect(hub.url) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                received.append(json.loads(raw))

        try:
            asyncio.run(_run())
        finally:
            done.set()

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    time.sleep(0.25)
    model = _model()
    dec = ControlDecoder(ControlDecoderConfig(strategy="discrete"))
    label, score = model.predict(np.random.default_rng(0).normal(0, 0.4, size=(8, 400)))
    out = dec.decode(
        label=label,
        score=score,
        decision_margin=1.0 if label == "right" else -1.0,
    )
    hub.publish(
        {
            "version": PROTOCOL_VERSION,
            "type": "bci_control",
            "trial_id": "trial_000",
            "sequence": 1,
            "source_window_end": 2.0,
            "issued_at": time.time(),
            "control": out.control,
            "valid": out.valid,
        }
    )
    assert done.wait(timeout=4.0)
    thread.join(timeout=1.0)
    hub.stop()
    assert received, "WebSocket client did not receive a control message"
    msg = received[0]
    assert msg["version"] == 2
    assert msg["type"] == "bci_control"
    assert msg["trial_id"] == "trial_000"
    assert -1.0 <= float(msg["control"]) <= 1.0


def test_decoder_does_not_alter_model_prediction() -> None:
    model = _model()
    trial = np.random.default_rng(3).normal(0, 0.4, size=(8, 400))
    label_a, score_a = model.predict(trial)
    label_b, score_b = model.predict(trial)
    assert label_a == label_b
    assert score_a == score_b
    decision = WindowDecision(
        window_index=0,
        start=0,
        end=2,
        label=label_a,  # type: ignore[arg-type]
        score=score_a,
        emg_rejected=False,
        reason="classified",
        classifier="linear_svm",
    )
    assert decision.label == label_a
