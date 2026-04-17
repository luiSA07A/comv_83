#!/bin/bash
# ============================================================
#  IDUN Slurm Job Script — MIL EfficientNet (Model 2)
# ============================================================

#SBATCH --job-name=odelia_mil
#SBATCH --account=share-ie-idi        
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00                 
#SBATCH --output=logs/%j_mil.out
#SBATCH --error=logs/%j_mil.err

echo "Starting MIL training at $(date)"

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ~/venvs/odelia/bin/activate

DATA_ROOT=/cluster/projects/vc/courses/TDT17/mic/ODELIA2025

python src/train.py \
    --data_root   $DATA_ROOT \
    --model       mil \
    --epochs      60 \
    --batch_size  4 \
    --lr          5e-5 \
    --output_dir  ./runs