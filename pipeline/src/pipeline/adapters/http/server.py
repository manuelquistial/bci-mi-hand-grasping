"""Threading HTTP server that stays quiet when clients close sockets early."""

from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

from pipeline.adapters.http.handler import make_handler
from pipeline.config import default_recordings_root


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Do not dump a traceback when the game closes a keep-alive socket."""

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    recordings_root: Path | None = None,
) -> None:
    """Block and serve the session API."""
    from pipeline.service import PipelineService

    root = Path(recordings_root) if recordings_root is not None else default_recordings_root()
    service = PipelineService(root)
    server = QuietThreadingHTTPServer((host, port), make_handler(service))
    print(f"pipeline serve http://{host}:{port}  recordings={root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        service.stop_recording()
        service.stop_online()
        server.server_close()
