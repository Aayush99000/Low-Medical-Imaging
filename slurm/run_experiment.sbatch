#!/bin/bash
#SBATCH --job-name=lowmed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Run one model on one dataset on a GPU node; any run_experiment.py arguments pass through. From the repo root:
#   sbatch slurm/run_experiment.sbatch --dataset pneumonia --model B_imagenet_probe
# One job per subset draw, in parallel (compare_models.py collects the draws afterwards):
#   sbatch --array=0-4 slurm/run_experiment.sbatch --dataset ham10000 --model C_medsiglip_finetune
# SigLIP / MedSigLIP fine-tuning needs more than a 16 GB V100; ask for a bigger GPU:
#   sbatch --gres=gpu:a100:1 slurm/run_experiment.sbatch ...
set -euo pipefail

ENV=${LOWMED_ENV:-/scratch/$USER/envs/lowmed}
export HF_HOME=/scratch/$USER/.cache/huggingface   # MedSigLIP token and weights (hf auth login with the same HF_HOME)
export TORCH_HOME=/scratch/$USER/.cache/torch      # torchvision ImageNet weights
cd "$SLURM_SUBMIT_DIR"

EXTRA=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    EXTRA+=(--subset_seeds "$SLURM_ARRAY_TASK_ID")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" run_experiment.py --num_workers "${SLURM_CPUS_PER_TASK:-4}" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
