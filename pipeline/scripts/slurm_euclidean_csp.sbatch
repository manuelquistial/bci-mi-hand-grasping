#!/bin/bash
# Cross-subject offline + single-subject online verification on one Slurm allocation.
#
# Resource estimate (shuqfa2024, 103 subjects, run 02):
#   offline-export ×2 : dump-trials all subjects, per-subject EA, CSP, LR/SVM
#                       ~30–90 min, ~8–20 GB peak with parallel dumps
#   online-replay  ×2 : one verification subject (~123 s @ 160 Hz)
#                       ~1–5 min, <2 GB
#   uv sync cold start: ~2–10 min
#
# Allocation (not exclusive): 8 CPUs, 32 GB, 4 h on `standard`.
# Only this job's cores/RAM are reserved; other users keep their slots.
#
# Submit from pipeline/ after `git pull` on the login node:
#   sbatch scripts/slurm_euclidean_csp.sbatch
# Optional:
#   ONLINE_SUBJECT=1 MAX_TRIALS=20 sbatch scripts/slurm_euclidean_csp.sbatch

#SBATCH --job-name=eucl-csp-xs
#SBATCH --account=parrarodriguez-eegpd
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=batch-results/slurm-%j.out
#SBATCH --error=batch-results/slurm-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p batch-results

ONLINE_SUBJECT="${ONLINE_SUBJECT:-1}"
DATASET="${DATASET:-shuqfa2024}"
MAX_TRIALS="${MAX_TRIALS:-}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} mem=32G"
echo "train=all_subjects online_subject=${ONLINE_SUBJECT} dataset=${DATASET} max_trials=${MAX_TRIALS:-all}"

if [[ -n "$MAX_TRIALS" ]]; then
  bash scripts/run_fsb_euclidean_csp.sh "$ONLINE_SUBJECT" "$DATASET" "$MAX_TRIALS"
else
  bash scripts/run_fsb_euclidean_csp.sh "$ONLINE_SUBJECT" "$DATASET"
fi
