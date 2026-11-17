#!/bin/bash
#SBATCH --job-name=lowmed-features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Cache one probe model's frozen features on one dataset; extract_features.py arguments pass through. From the repo root:
#   sbatch slurm/extract_features.sbatch --dataset chexpert --model C_medsiglip_probe
set -euo pipefail

ENV=${LOWMED_ENV:-/scratch/$USER/envs/lowmed}
export HF_HOME=/scratch/$USER/.cache/huggingface   # MedSigLIP / SigLIP weights
export TORCH_HOME=/scratch/$USER/.cache/torch      # torchvision ImageNet weights
cd "$SLURM_SUBMIT_DIR"
# HF_TOKEN for the gated MedSigLIP weights, from .env in the repo root (gitignored; chmod 600)
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" extract_features.py --num_workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
