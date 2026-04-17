#!/bin/bash
# ============================================================
#  IDUN Slurm Job Script — MIL EfficientNet (Model 2)
#  Submit with:  sbatch scripts/train_mil.sh
# ============================================================

#SBATCH --job-name=odelia_mil
#SBATCH --account=share-ie-idi          # <-- Check if your group has a specific account
#SBATCH --partition=GPUQ                # Request a GPU node 
#SBATCH --gres=gpu:1                    # Use 1 GPU
#SBATCH --cpus-per-task=8               # CPU cores for data loading
#SBATCH --mem=32G                       # 32GB RAM [cite: 45]
#SBATCH --time=10:00:00                 # 10 hours max run time [cite: 41]
#SBATCH --output=logs/%j_mil.out        # Log file name (e.g., 12345_mil.out)
#SBATCH --error=logs/%j_mil.err         # Error file name

echo "Job ID: $SLURM_JOB_ID"
date

# 1. Load the cluster software
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# 2. Activate the virtual environment you created with setup_idun.sh
source ~/venvs/odelia/bin/activate

# 3. Path to the official dataset on IDUN 
DATA_ROOT=/cluster/projects/vc/courses/TDT17/mic/ODELIA2025

# 4. Run the training script
python src/train.py \
    --data_root   $DATA_ROOT \
    --model       mil \
    --epochs      60 \
    --batch_size  4 \
    --lr          5e-5 \
    --in_channels 1 \
    --spatial_size 96 96 32 \
    --output_dir  ./runs

echo "Done at $(date)"