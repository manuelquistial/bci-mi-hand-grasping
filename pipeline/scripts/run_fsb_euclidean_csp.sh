#!/usr/bin/env bash
# Offline (euclidean + CSP + LR/SVM) → register → online-sim on the same host.
#
# Usage (from the pipeline/ directory on FSB):
#   bash scripts/run_fsb_euclidean_csp.sh [SUBJECT] [DATASET] [MAX_TRIALS]
#
# Defaults: first listed subject via offline-export, dataset=shuqfa2024, all trials.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SUBJECT="${1:-}"
DATASET="${2:-shuqfa2024}"
MAX_TRIALS="${3:-}"
CONFIG="${ROOT}/configs/fsb-euclidean-csp.json"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${ROOT}/batch-results/fsb-euclidean-csp-${STAMP}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "error: missing $CONFIG" >&2
  exit 1
fi

# Rewrite subject/dataset into a working config so both modes match this run.
WORK_CONFIG="${OUT_DIR}/modes.json"
python - <<PY
import json
from pathlib import Path
src = Path(${CONFIG@Q})
dst = Path(${WORK_CONFIG@Q})
payload = json.loads(src.read_text())
subject = ${SUBJECT@Q}
dataset = ${DATASET@Q}
for mode in payload["modes"]:
    mode["dataset"] = dataset
    if subject:
        mode["subject"] = subject
    else:
        mode["subject"] = None
dst.write_text(json.dumps(payload, indent=2) + "\n")
print(dst)
PY

echo "==> Offline export (euclidean + CSP + logistic_regression | linear_svm)"
uv run pipeline offline-export --config "$WORK_CONFIG" | tee "$OUT_DIR/offline-export.log"

echo "==> Registered packs"
uv run pipeline offline-list --dataset "$DATASET" | tee "$OUT_DIR/offline-list.txt"

run_online () {
  local name="$1"
  local log="$OUT_DIR/online-${name}.jsonl"
  local summary="$OUT_DIR/online-${name}-summary.json"
  echo "==> Online sim for registry name=${name}"
  local args=(online-sim --registry "$name" --log "$log")
  if [[ -n "$MAX_TRIALS" ]]; then
    args+=(--max-trials "$MAX_TRIALS")
  fi
  uv run pipeline "${args[@]}" | tee "$OUT_DIR/online-${name}.log"
  # Last JSON object printed by online-sim is the summary; keep the whole log too.
  cp "$log" "$OUT_DIR/" 2>/dev/null || true
  echo "online log: $log"
}

run_online "euclidean-csp-logistic_regression"
run_online "euclidean-csp-linear_svm"

echo "==> Done. Results under $OUT_DIR"
ls -la "$OUT_DIR"
