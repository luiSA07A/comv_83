#!/bin/bash
#SBATCH --job-name=odelia_predict
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prediction_%j.out

# 1. LOAD THE SYSTEM PYTHON FIRST (This is the missing piece)
module load Python/3.11.3-GCCcore-12.3.0

# 2. Activate your environment
source ~/venvs/odelia/bin/activate

# 3. Add this to be extra safe (helps find those .so files)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_ROOT/lib

# Run Prediction for DenseNet
echo "Running DenseNet Predictions..."
python src/predict.py \
  --model densenet \
  --checkpoint runs/best_model_densenet.pt \
  --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025 \
  --subset val \
  --output runs/preds_densenet.csv

# Run Prediction for MIL
echo "Running MIL Predictions..."
python src/predict.py \
  --model mil \
  --checkpoint runs/best_model_mil.pt \
  --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025 \
  --subset val \
  --output runs/preds_mil.csv