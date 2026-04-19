#!/bin/bash
# ============================================================
#  IDUN Slurm Job Script — DenseNet121 (Model 1)
# ============================================================

#SBATCH --job-name=odelia_densenet
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ               
#SBATCH --gres=gpu:1                   
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                  
#SBATCH --time=12:00:00               
#SBATCH --output=logs/%j_densenet.out
#SBATCH --error=logs/%j_densenet.err

echo "Starting DenseNet training at $(date)"

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate environment
source ~/venvs/odelia/bin/activate

# Official IDUN path for ODELIA 2025 
DATA_ROOT=/cluster/projects/vc/courses/TDT17/mic/ODELIA2025

python src/train.py \
    --data_root   $DATA_ROOT \
    --model       densenet \
    --epochs      100 \
    --batch_size  4 \
    --lr          5e-5 \
    --output_dir  ./runs/mil_v2