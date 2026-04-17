#!/bin/bash
# ============================================================
#  IDUN Slurm Job Script — DenseNet121 (Model 1)
#  Submit with:  sbatch train_densenet.sh
# ============================================================

#SBATCH --job-name=odelia_densenet
#SBATCH --account=YOUR_ACCOUNT         # <-- replace with your NTNU project account
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00                # 8 hours max
#SBATCH --output=logs/%j_densenet.out
#SBATCH --error=logs/%j_densenet.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
date

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# ── Activate virtual environment ──────────────────────────────────────────────
source ~/venvs/odelia/bin/activate

# ── Run training ──────────────────────────────────────────────────────────────
DATA_ROOT=/cluster/projects/vc/courses/TDT17/mic/ODELIA2025

python src/train.py \
    --data_root   $DATA_ROOT \
    --model       densenet \
    --epochs      50 \
    --batch_size  4 \
    --lr          1e-4 \
    --in_channels 1 \
    --spatial_size 96 96 32 \
    --output_dir  ./runs

echo "Done at $(date)"
