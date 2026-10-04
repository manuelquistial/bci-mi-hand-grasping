"""Read samples from an LSL StreamInlet."""

from __future__ import annotations

from typing import Any, Iterator

from pipeline.discover import discover_streams, summarize_stream
from pipeline.models import SampleChunk, StreamSummary


class LslStreamReader:
    """
    Thin wrapper around ``pylsl.StreamInlet``.

    Typical use::

        with LslStreamReader.open(stream_type="EEG") as reader:
            chunk = reader.pull_chunk(timeout=1.0)
            print(chunk.samples)
    """

    def __init__(self, inlet: Any, summary: StreamSummary) -> None:
        self._inlet = inlet
        self.summary = summary

    @classmethod
    def open(
        cls,
        *,
        name: str | None = None,
        stream_type: str | None = None,
        wait_seconds: float = 5.0,
        max_buflen: int = 360,
        max_chunklen: int = 0,
        recover: bool = True,
    ) -> LslStreamReader:
        """Discover a stream, open an inlet, and return a reader."""
        from pylsl import StreamInlet, resolve_byprop, resolve_streams

        if name:
            infos = resolve_byprop("name", name, minimum=1, timeout=wait_seconds)
        elif stream_type:
            infos = resolve_byprop("type", stream_type, minimum=1, timeout=wait_seconds)
        else:
            infos = resolve_streams(wait_time=wait_seconds)

        if not infos:
            raise TimeoutError(
                "No LSL streams found. "
                "Start a source (e.g. simulator/simopenbci) and retry."
            )

        info = infos[0]
        if name and info.name() != name:
            raise TimeoutError(f"Stream named {name!r} not found within {wait_seconds}s.")

        inlet = StreamInlet(
            info,
            max_buflen=max_buflen,
            max_chunklen=max_chunklen,
            recover=recover,
        )
        full_info = inlet.info()
        return cls(inlet, summarize_stream(full_info))

    @classmethod
    def from_discovered(
        cls,
        summary: StreamSummary,
        *,
        wait_seconds: float = 5.0,
        **inlet_kwargs: Any,
    ) -> LslStreamReader:
        """Open a reader for a previously discovered stream (by name)."""
        return cls.open(name=summary.name, wait_seconds=wait_seconds, **inlet_kwargs)

    def pull_sample(self, timeout: float = 1.0) -> tuple[list[float], float] | None:
        """Pull a single sample. Returns ``None`` on timeout."""
        sample, timestamp = self._inlet.pull_sample(timeout=timeout)
        if sample is None:
            return None
        return list(sample), float(timestamp)

    def pull_chunk(
        self,
        timeout: float = 0.0,
        max_samples: int = 1024,
    ) -> SampleChunk:
        """Pull up to ``max_samples`` available samples."""
        samples, timestamps = self._inlet.pull_chunk(
            timeout=timeout,
            max_samples=max_samples,
        )
        coerced: list[list[float] | list[str]] = []
        for s in samples:
            if isinstance(s, str):
                coerced.append([s])
            else:
                coerced.append(list(s))
        return SampleChunk(
            samples=coerced,  # type: ignore[arg-type]
            timestamps=[float(t) for t in timestamps],
        )

    def iter_chunks(
        self,
        *,
        timeout: float = 1.0,
        max_samples: int = 1024,
        max_chunks: int | None = None,
    ) -> Iterator[SampleChunk]:
        """Yield chunks until interrupted or ``max_chunks`` is reached."""
        yielded = 0
        while max_chunks is None or yielded < max_chunks:
            chunk = self.pull_chunk(timeout=timeout, max_samples=max_samples)
            if not chunk.samples:
                continue
            yielded += 1
            yield chunk

    def close(self) -> None:
        self._inlet = None

    def __enter__(self) -> LslStreamReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_streams(
    *,
    stream_type: str | None = None,
    name: str | None = None,
    wait_seconds: float = 2.0,
) -> list[StreamSummary]:
    """Public alias used by the CLI and callers."""
    return discover_streams(stream_type=stream_type, name=name, wait_seconds=wait_seconds)
