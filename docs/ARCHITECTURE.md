# Architecture

This repository holds three cooperating packages that together run an EEG motor-imagery BCI game. This page is the entry point for how they fit together and which rules CI enforces.

## Packages

| Package | Language | Role |
|---------|----------|------|
| `pipeline/` | Python 3.11 (`uv`) | Records and classifies EEG, trains and serves online models, exposes the HTTP API and two WebSocket feeds. |
| `game/` | TypeScript (Next.js + Phaser) | Browser UI: the coin-collector game, settings, and the `/monitor` page. Never opens LSL itself. |
| `simulator/` | Python 3.11 (`uv`) | `simopenbci`: replays dataset or recorded EEG to LSL and emits `*_Markers`. |

```
simulator ──LSL (EEG + Markers)──▶ pipeline ──HTTP :8765──────▶ game
                                      │      ──control WS :8090─▶ game (/game)
                                      └──────  observe WS :8091─▶ game (/monitor)
```

The pipeline starts the simulator as a subprocess (`pipeline.adapters.simulator`) and talks to it only through LSL and its CLI.

## Named ports

| Port | Name | Constant / source | Direction |
|------|------|-------------------|-----------|
| `8765` | HTTP API | `pipeline serve --port` default, `pipeline.adapters.http.server` | game → pipeline |
| `8090` | control WS | `pipeline.config.CONTROL_WS_PORT` | pipeline → game |
| `8091` | observe WS | `pipeline.config.OBSERVE_WS_PORT` | pipeline → monitor |

The two WebSocket ports are asserted by `pipeline/tests/test_architecture.py`. The game defaults to `http://127.0.0.1:8765` (`game/src/types/settings/pipeline.ts`).

## Pipeline: hexagonal layers

```
domain  ←  application  ←  adapters
```

Dependencies point inward: an outer layer may import an inner one, never the reverse.

| Layer | Location (`pipeline/src/pipeline/`) | Contents |
|-------|-------------------------------------|----------|
| domain | `online/`, `features/`, `spatial/`, `alignment/`, `classify/`, plus top-level signal modules (`preprocess.py`, `epochs.py`, `windows.py`, `params.py`, `models.py`, …) | Pure signal processing, decoding, gating, and wire-message builders. No HTTP, no sockets, no process spawning. |
| application | `application/` | Use cases: online session lifecycle, hubs, recording, cohort/dataset/model catalogs, batch runner, observe presenter. |
| adapters | `adapters/{http,ws,lsl,simulator,storage,cli}/` | HTTP server and handler, WebSocket hub, LSL streams, simulator subprocess/catalog, batch file storage, CLI commands. |

`service.py` is the composition root for the HTTP service and sits outside the layers: it wires adapters and application services together.

Known debt: some `application/` modules still import concrete `adapters.*` helpers (for example `adapters.simulator.catalog`, `adapters.ws.hub`). The direction target is for them to depend on ports instead. This is not gated yet.

## Dependency rules

1. **No `online` → `service`.** No module under `pipeline/src/pipeline/online/` imports `pipeline.service`. Enforced by `scripts/check_architecture.py` and `pipeline/tests/test_architecture.py`.
2. **Inner layers never import outer layers.** Domain does not import `application` or `adapters`.
3. **The game talks to the pipeline over HTTP and WebSocket only.** It never imports Python, reads LSL, or reads pipeline files directly.
4. **The simulator exposes `PlaybackService` plus ports.** The core (`simopenbci.PlaybackService`) depends on `SampleSink`, `Notifier`, and `Clock` (`simopenbci/ports.py`). Qt and pylsl live in adapters (`sinks/`, `notifiers/`, `gui.py`). Dataset formats live in `dataset_loaders/`.
5. **Pipeline and simulator share no Python imports.** They share only the marker wire format and LSL stream names.

## Line budget

Every owned production `.py`, `.ts`, and `.tsx` file must be **≤ 300 lines**.

Excluded: tests (`tests/`, `test_*.py`, `conftest.py`, `*.test.ts(x)`), `venv`/`.venv`, `node_modules`, `__pycache__`, `.next`, `dist`, `build`, and local data folders. `__init__.py` files are counted by the repo-wide gate.

## Contracts

| Contract | Version | Fixture | Frozen by |
|----------|---------|---------|-----------|
| Marker text `kind:code:label` | n/a | [`contracts/markers.json`](../contracts/markers.json) | `simulator/tests/test_marker_contract.py`, `pipeline/tests/test_marker_wire_contract.py` |
| Control WS (`bci_control`, `trial_start`) | **v2** | [`contracts/control_v2.json`](../contracts/control_v2.json) | `pipeline/tests/test_contract_schemas.py` |
| Observe WS (`pipeline_stage`, `stream_frame`) | **v1** | [`contracts/observe_v1.json`](../contracts/observe_v1.json) | `pipeline/tests/test_contract_schemas.py` |

### Marker contract: `kind:code:label`

Each marker is one string sample on the LSL `*_Markers` stream:

- `kind`: event class (`trial_start`, `rest`, …).
- `code`: dataset-native code (`2`, `T1`, `769`, …).
- `label`: short display label (`left`, `right`, `rest`). It may contain `:`; everything after the second separator belongs to the label.

Producer: `simulator` (`TrialEvent.marker_string`, `simopenbci/markers_contract.py`). Consumer: `pipeline.live.parse_marker_text`, which uses `kind` and `label` and ignores `code`. The game posts the same three fields as JSON to `/v1/record/marker` and `/v1/online/marker`.

### Control v2 and observe v1

Both are JSON messages with a top-level `version` and `type`. Changing a key set requires a version bump and an update to the matching file in `contracts/`.

## CI gates

`scripts/check_architecture.sh` runs `scripts/check_architecture.py` (stdlib only, works offline) and fails on a line-budget or `online → service` violation. `.github/workflows/architecture.yml` runs it first, then the pipeline architecture and contract tests, the simulator tests, and the game tests.

```bash
scripts/check_architecture.sh               # offline gates only
scripts/check_architecture.sh --with-tests  # plus pipeline, simulator, game test suites
```

For a manual end-to-end walkthrough see [`scripts/e2e_checklist.md`](../scripts/e2e_checklist.md).
