#!/bin/bash
#SBATCH --job-name=lowmed-queue
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
# Feed GPU jobs to Slurm as the gpu QOS allows (on Explorer: 8 submitted / 4 running per user, and each
# array task counts as one job). Runs on a CPU node, so nothing has to stay open on the login node.
# Each non-comment line of the queue file is the argument list of one sbatch call, e.g.
#   --time=04:00:00 slurm/run_experiment.sbatch --dataset pneumonia --model B_imagenet_probe --subset_seeds 0
# From the repo root:
#   sbatch slurm/submit_queue.sh logs/queue.txt
set -uo pipefail

QUEUE=${1:?usage: sbatch slurm/submit_queue.sh <queue file>}
MAX_SUBMITTED=${MAX_SUBMITTED:-8}
POLL_SECONDS=${POLL_SECONDS:-120}
cd "$SLURM_SUBMIT_DIR"

n_gpu_jobs() { squeue -u "$USER" -h -r -q gpu -o %i | wc -l; }

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    read -ra args <<< "$line"
    until (( $(n_gpu_jobs) < MAX_SUBMITTED )) && job=$(sbatch --parsable "${args[@]}" 2>/dev/null); do
        sleep "$POLL_SECONDS"
    done
    echo "$(date '+%F %T') submitted $job: $line"
done < "$QUEUE"
echo "$(date '+%F %T') queue empty"
