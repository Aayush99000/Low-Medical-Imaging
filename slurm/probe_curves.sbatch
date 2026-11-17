#!/bin/bash
#SBATCH --job-name=lowmed-curves
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Probe data-efficiency curves from the cached features (CPU only); probe_curves.py arguments pass through.
# Run after slurm/extract_features.sbatch has cached every probe model for the dataset. From the repo root:
#   sbatch slurm/probe_curves.sbatch --dataset chexpert
set -euo pipefail

ENV=${LOWMED_ENV:-/scratch/$USER/envs/lowmed}
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg

"$ENV/bin/python" probe_curves.py --n_jobs "${SLURM_CPUS_PER_TASK:-4}" "$@"
