"""Start/stop the control and observe WebSocket hubs with rollback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.adapters.ws.hub import JsonWsHub
from pipeline.config import CONTROL_WS_PORT, OBSERVE_WS_PORT


@dataclass(frozen=True)
class HubPair:
    control: JsonWsHub
    observe: JsonWsHub


def resolve_observe_port(control_port: int, observe_ws_port: int | None) -> int:
    if observe_ws_port is not None:
        return int(observe_ws_port)
    if int(control_port) == CONTROL_WS_PORT:
        return OBSERVE_WS_PORT
    return int(control_port) + 1


def start_hub_pair(
    *,
    control_port: int = CONTROL_WS_PORT,
    observe_port: int | None = None,
) -> HubPair:
    """Bind control then observe; stop control if observe fails."""
    control = JsonWsHub(port=int(control_port))
    observe: JsonWsHub | None = None
    try:
        control.start()
        observe = JsonWsHub(port=resolve_observe_port(control_port, observe_port))
        observe.start()
    except Exception:
        if observe is not None:
            observe.stop()
        control.stop()
        raise
    return HubPair(control=control, observe=observe)


def stop_hub_pair(hubs: HubPair | None) -> None:
    if hubs is None:
        return
    hubs.control.stop()
    hubs.observe.stop()


def online_status_urls(hubs: HubPair | None) -> dict[str, Any]:
    if hubs is None:
        return {"control_ws_url": None, "observe_ws_url": None}
    return {
        "control_ws_url": hubs.control.url,
        "observe_ws_url": hubs.observe.url,
    }
