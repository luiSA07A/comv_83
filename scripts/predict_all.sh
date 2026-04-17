#!/bin/bash
#SBATCH --job-name=odelia_predict
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prediction_%j.out

# 1. Load Python
module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/odelia/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_ROOT/lib

# 2. Run Prediction for DenseNet
# We use --subset val because that is what is in your metadata
echo "Running DenseNet Predictions..."
python src/predict.py \
  --model densenet \
  --checkpoint runs/best_model_densenet.pt \
  --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025 \
  --subset val \
  --output runs/preds_densenet.json

# 3. Run Prediction for MIL
echo "Running MIL Predictions..."
python src/predict.py \
  --model mil \
  --checkpoint runs/best_model_mil.pt \
  --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025 \
  --subset val \
  --output runs/preds_mil.json