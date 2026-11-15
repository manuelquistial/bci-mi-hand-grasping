# E2E checklist: pipeline, monitor, simulator, coin success path

Manual smoke test for the online simulated path. Run `scripts/e2e_smoke.sh` to print these steps.

Notes:

- The monitor button was previously labelled **Abrir**; it is now **Open monitor**.
- Ports: HTTP `8765`, control WS `8090`, observe WS `8091` (see [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)).
- The simulator is started by the pipeline (`POST /v1/online/start`) when the game's **Start session** button is pressed. You do not launch it by hand in this flow.

## Prerequisites

- [ ] `uv` installed; `uv sync --all-groups` done in `pipeline/` and `simulator/`.
- [ ] `npm ci` done in `game/`.
- [ ] liblsl available for `pylsl` (macOS: see `simulator/README.md`).
- [ ] A local dataset under `simulator/data/<dataset_id>/` (for example `shuqfa2024`).
- [ ] An online model available in the registry (`pipeline/online-models/`).

## Steps

1. **Start the pipeline**
   ```bash
   cd pipeline && uv run pipeline serve
   ```
   - [ ] Service listens on `http://127.0.0.1:8765`.
   - [ ] `curl http://127.0.0.1:8765/v1/status` returns JSON.

2. **Start the game**
   ```bash
   cd game && npm run dev
   ```
   - [ ] Open `http://localhost:3000` and choose **Online simulated** (`/game?step=simulator`).

3. **Configure the session**
   - [ ] Pipeline base URL is `http://127.0.0.1:8765` and control port is `8090`.
   - [ ] Pick a dataset, subject, and online model.

4. **Open monitor**
   - [ ] Click **Open monitor**. A new tab opens at `/monitor?baseUrl=…&controlPort=…`.
   - [ ] The monitor shows it is connected (observe WS `ws://127.0.0.1:8091`) and waiting for data.

5. **Start simulator**
   - [ ] Back in the game tab, click **Start session**.
   - [ ] The pipeline launches the simulator; `GET /v1/status` shows `online.running: true` and `online.samples_seen` increasing.
   - [ ] The simulator emits LSL EEG on `OpenBCI_EEG` plus the `*_Markers` stream (`kind:code:label`).

6. **Expect stage frames**
   - [ ] The monitor receives `stream_frame` messages continuously (raw and filtered traces move).
   - [ ] During each trial it receives `pipeline_stage` messages with `trial_id`, `cue`, `p_left`/`p_right`, `strategy`, and `gate`.
   - [ ] `GET /v1/status` shows `last_pipeline_stage` and `last_stream_frame` populated.

7. **Expect the coin success path**
   - [ ] The game shows the cue and the coin on the cued side.
   - [ ] Control WS (`8090`) delivers `trial_start` then `bci_control` messages; the character walks toward the coin.
   - [ ] On a correct decision the character reaches the coin, plays the success pose, and the trial outcome is **success**.
   - [ ] The next trial starts after the inter-trial interval.

8. **Stop**
   - [ ] Press space in the game tab to stop the session.
   - [ ] `GET /v1/status` shows `online.running: false`; the simulator process is gone.
   - [ ] The session appears under **Sessions & export** (`/results`).

## Automated checks to run first

```bash
scripts/check_architecture.sh --with-tests
```

## Failure triage

| Symptom | Likely cause |
|---------|--------------|
| Monitor stays empty | Observe WS `8091` not reachable, or the online run has not started. |
| Game does not move | Control WS `8090` not connected, or the model returned `valid: false`. |
| No samples in `/v1/status` | Simulator did not start or LSL name mismatch (expected `OpenBCI_EEG`). |
| No markers | Dataset has no events sidecar, or `--no-markers` was used. |
