#!/bin/bash
#SBATCH --job-name=lowmed-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# One-time environment build on Northeastern's Explorer cluster. From the repo root:
#   mkdir -p logs && sbatch slurm/setup_env.sh
# The env and the conda / pip caches go to /scratch: the 75 GB home quota is too small for them.
# Installing torch onto /scratch takes a while; rerunning resumes an env that a timed-out job left behind.
set -euo pipefail

ENV=${LOWMED_ENV:-/scratch/$USER/envs/lowmed}
export CONDA_PKGS_DIRS=/scratch/$USER/.cache/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip

module load miniconda3/25.9.1
if [[ ! -x "$ENV/bin/python" ]]; then
    conda create -y -p "$ENV" -c conda-forge --override-channels python=3.11 pip
fi
# CUDA 12.6 wheels are built for the V100s (sm_70) as well as the A100s and H200s
"$ENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu126
"$ENV/bin/pip" install -r requirements.txt
"$ENV/bin/python" -c "import torch, torchvision, transformers, cv2; print(torch.__version__, torchvision.__version__, transformers.__version__, cv2.__version__)"
