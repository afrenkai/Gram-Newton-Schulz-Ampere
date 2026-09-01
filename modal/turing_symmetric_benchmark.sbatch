#!/bin/bash
#SBATCH --job-name=gns-sym-bench
#SBATCH --partition=quick
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --output=/scratch/afrenk/gns-ampere-email-benchmark/sym-bench-%j.out
#SBATCH --error=/scratch/afrenk/gns-ampere-email-benchmark/sym-bench-%j.err
set -euo pipefail
source /etc/profile
module load cuda/12.9.0/b3yzg3f
cd /scratch/afrenk/gns-ampere-email-benchmark
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.current.sm,clocks.current.memory --format=csv,noheader
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions"
.venv/bin/python modal/benchmark_symmetric_cutlass.py \
  --batch-sizes 1 8 32 \
  --warmups 5 \
  --repeats 20 \
  --output "symmetric-cutlass-results-${SLURM_JOB_ID}.json"
