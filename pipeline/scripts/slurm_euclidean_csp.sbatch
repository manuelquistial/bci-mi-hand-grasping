#!/bin/bash
# Offline + online (euclidean / CSP / LR|SVM) on one Slurm allocation.
#
# Resource estimate (shuqfa2024 one subject, SIG_02 ≈ 123 s @ 160 Hz, ~4 MB):
#   offline-export ×2 : dump-trials + CSP + LR/SVM          ~1–5 min, <2 GB
#   online-replay  ×2 : full file through causal pipeline   ~1–5 min, <2 GB
#   uv sync cold start                                     ~2–10 min
# Totals with margin: 4 CPUs, 8 GB RAM, 60 min on `dev`.
# Not exclusive: only this allocation; other jobs keep their own slots.
#
# Submit from the pipeline/ directory after `git pull` on the login node:
#   sbatch scripts/slurm_euclidean_csp.sbatch
# Optional env overrides:
#   SUBJECT=1 DATASET=shuqfa2024 MAX_TRIALS=20 sbatch scripts/slurm_euclidean_csp.sbatch

#SBATCH --job-name=eucl-csp-online
#SBATCH --account=parrarodriguez-eegpd
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=batch-results/slurm-%j.out
#SBATCH --error=batch-results/slurm-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p batch-results

SUBJECT="${SUBJECT:-1}"
DATASET="${DATASET:-shuqfa2024}"
MAX_TRIALS="${MAX_TRIALS:-}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} mem=8G"
echo "subject=${SUBJECT} dataset=${DATASET} max_trials=${MAX_TRIALS:-all}"

if [[ -n "$MAX_TRIALS" ]]; then
  bash scripts/run_fsb_euclidean_csp.sh "$SUBJECT" "$DATASET" "$MAX_TRIALS"
else
  bash scripts/run_fsb_euclidean_csp.sh "$SUBJECT" "$DATASET"
fi
