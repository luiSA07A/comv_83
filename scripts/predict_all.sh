#!/bin/bash
#SBATCH --job-name=odelia_predict
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prediction_%j.out

# Activate your environment
source ~/venvs/odelia/bin/activate

# 1. Run Prediction for DenseNet
echo "Running DenseNet Predictions..."
python src/predict.py \
  --model densenet \
  --model_path runs/best_model_densenet.pt \
  --output_csv runs/preds_densenet.csv

# 2. Run Prediction for MIL
echo "Running MIL Predictions..."
python src/predict.py \
  --model mil \
  --model_path runs/best_model_mil.pt \
  --output_csv runs/preds_mil.csv
