#!/bin/bash
# ============================================================
#  IDUN Slurm Job Script — DenseNet121 (Model 1)
# ============================================================

#SBATCH --job-name=odelia_densenet
#SBATCH --account=share-ie-idi          # Replace with your group's account if different
#SBATCH --partition=GPUQ                # Request GPU partition [cite: 39]
#SBATCH --gres=gpu:1                    # Request 1 GPU [cite: 37]
#SBATCH --cpus-per-task=8               # Cores for data processing
#SBATCH --mem=32G                       # Request 32GB RAM [cite: 37]
#SBATCH --time=12:00:00                 # 12 hours max time [cite: 41]
#SBATCH --output=logs/%j_densenet.out
#SBATCH --error=logs/%j_densenet.err

echo "Starting DenseNet training at $(date)"

# Load modules [cite: 37]
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate environment
source ~/venvs/odelia/bin/activate

# Official IDUN path for ODELIA 2025 [cite: 96]
DATA_ROOT=/cluster/projects/vc/courses/TDT17/mic/ODELIA2025

python src/train.py \
    --data_root   $DATA_ROOT \
    --model       densenet \
    --epochs      50 \
    --batch_size  4 \
    --lr          1e-4 \
    --output_dir  ./runs