#!/bin/bash
#SBATCH --job-name=gns-cutlass-exp
#SBATCH --partition=quick
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --output=/scratch/afrenk/gns-ampere-email-benchmark/cutlass-experiment-%j.out
#SBATCH --error=/scratch/afrenk/gns-ampere-email-benchmark/cutlass-experiment-%j.err
set -euo pipefail
source /etc/profile
module load cuda/12.9.0/b3yzg3f
cd /scratch/afrenk/gns-ampere-email-benchmark
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions"
mode="${1:?expected dimension, mirror, cold-cache, or power}"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.current.sm,clocks.current.memory,power.limit --format=csv,noheader
case "$mode" in
    dimension)
        .venv/bin/python modal/benchmark_dimension_sweep.py             --warmups 3             --repeats 10             --output "dimension-sweep-${SLURM_JOB_ID}.json"
        ;;
    mirror)
        .venv/bin/python modal/benchmark_fused_mirror.py             --batch-sizes 1 8             --warmups 5             --repeats 20             --output "fused-mirror-${SLURM_JOB_ID}.json"
        ;;
    cold-cache)
        cache_root="$PWD/cold-cache-${SLURM_JOB_ID}"
        rm -rf "$cache_root"
        .venv/bin/python modal/benchmark_cold_cache.py             --cold-trials 2             --warm-process-trials 2             --cache-root "$cache_root"             --output "cold-cache-${SLURM_JOB_ID}.json"
        ;;
    power)
        sample_directory="$PWD/power-samples-${SLURM_JOB_ID}"
        mkdir -p "$sample_directory"
        .venv/bin/python modal/benchmark_power_utilization.py             --duration-seconds 5             --idle-seconds 2             --sample-interval-ms 50             --sample-directory "$sample_directory"             --output "power-utilization-${SLURM_JOB_ID}.json"
        ;;
    *)
        echo "unknown mode: $mode" >&2
        exit 2
        ;;
esac
