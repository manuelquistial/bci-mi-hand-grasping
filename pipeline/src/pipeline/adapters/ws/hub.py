"""Broadcast JSON messages to every connected WebSocket client."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from pipeline.config import CONTROL_WS_PORT


class JsonWsHub:
    """Thread-backed WebSocket fan-out used by control and observe publishers."""

    def __init__(self, host: str = "127.0.0.1", port: int = CONTROL_WS_PORT) -> None:
        self.host = host
        self.port = int(port)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._clients: set[Any] = set()
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name=f"ws-{self.port}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise TimeoutError(f"WebSocket failed to bind on {self.url}")
        if self._error:
            raise RuntimeError(self._error)

    def stop(self) -> None:
        loop = self._loop
        stop = self._stop
        if loop is not None and stop is not None and loop.is_running():
            loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._stop = None
        self._clients.clear()

    def publish(self, message: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except OSError as exc:
            self._error = str(exc)
            self._ready.set()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()

        async def handler(websocket: Any) -> None:
            self._clients.add(websocket)
            try:
                await websocket.wait_closed()
            except ConnectionClosed:
                pass
            finally:
                self._clients.discard(websocket)

        try:
            async with serve(handler, self.host, self.port):
                self._ready.set()
                assert self._stop is not None
                await self._stop.wait()
        except OSError as exc:
            self._error = str(exc)
            self._ready.set()
            return

        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass

    async def _broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(message)
        dead: list[Any] = []
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:  # noqa: BLE001
                dead.append(client)
        for client in dead:
            self._clients.discard(client)


# Backward-compatible name used by existing tests and LiveControlBridge.
ControlWsHub = JsonWsHub
