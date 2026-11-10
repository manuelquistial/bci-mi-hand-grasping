#!/usr/bin/env bash
# Offline (euclidean + CSP + LR | SVM) → register → online-replay (no LSL).
#
# Usage (from pipeline/ on FSB, preferably inside a Slurm allocation):
#   bash scripts/run_fsb_euclidean_csp.sh [SUBJECT] [DATASET] [MAX_TRIALS]
#
# Defaults: SUBJECT=1, DATASET=shuqfa2024, all hand trials.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

SUBJECT="${1:-1}"
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

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv into ~/.local"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> Syncing pipeline + simulator envs"
uv sync --all-groups
uv sync --project ../simulator --all-groups

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
    mode["subject"] = subject or None
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
  echo "==> Online replay for registry name=${name}"
  local args=(online-replay --registry "$name" --dataset "$DATASET" --subject "$SUBJECT" --log "$log")
  if [[ -n "$MAX_TRIALS" ]]; then
    args+=(--max-trials "$MAX_TRIALS")
  fi
  uv run pipeline "${args[@]}" | tee "$OUT_DIR/online-${name}.log"
  echo "online log: $log"
}

run_online "euclidean-csp-logistic_regression"
run_online "euclidean-csp-linear_svm"

echo "==> Done. Results under $OUT_DIR"
ls -la "$OUT_DIR"
