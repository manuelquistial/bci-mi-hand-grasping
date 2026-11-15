#!/usr/bin/env bash
# Prints the manual E2E smoke steps. It does not start any process.
set -euo pipefail

cat <<'EOF'
E2E smoke (online simulated path) - full checklist: scripts/e2e_checklist.md
Note: the monitor button "Abrir" was renamed to "Open monitor".

1. Start pipeline : (cd pipeline && uv run pipeline serve)          # HTTP :8765
2. Start game     : (cd game && npm run dev)                        # http://localhost:3000
3. In the game    : Online simulated -> pick dataset/subject/model
4. Open monitor   : click "Open monitor" (new tab, observe WS :8091)
5. Start simulator: click "Start session" (pipeline spawns the simulator)
6. Expect         : stream_frame + pipeline_stage frames in the monitor
7. Expect         : trial_start/bci_control on control WS :8090, character
                    reaches the coin, outcome = success
8. Stop           : press space; GET /v1/status shows online.running=false
EOF
