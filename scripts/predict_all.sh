#!/bin/bash
#SBATCH --job-name=RSH_Leaderboard_All
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/rsh_full_run_%j.out

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/odelia/bin/activate

# Paths to RSH specific data
DATA_DIR="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/data_unilateral"
SPLIT_FILE="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/metadata_unilateral/split.csv"

# --- 1. RUN DENSENET ---
echo "Running DenseNet on RSH Cohort..."
python src/predict.py \
  --model densenet \
  --checkpoint runs/mil_v2/best_model_densenet.pt \
  --data_root $DATA_DIR \
  --split_file $SPLIT_FILE \
  --output_csv predictions_densenet_leaderboard.csv

# --- 2. RUN MIL ---
echo "Running MIL on RSH Cohort..."
python src/predict.py \
  --model mil \
  --checkpoint runs/densenet_v2/best_model_mil.pt \
  --data_root $DATA_DIR \
  --split_file $SPLIT_FILE \
  --output_csv predictions_mil_leaderboard.csv

echo "Inference Complete. You have two files to choose from:"
ls -lh predictions_*_leaderboard.csv