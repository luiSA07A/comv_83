#!/bin/bash
#SBATCH --job-name=odelia_predict
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prediction_%j.out

# 1. Load the specific Python module used on IDUN
module load Python/3.11.3-GCCcore-12.3.0

# 2. Activate your virtual environment
source ~/venvs/odelia/bin/activate

# 3. Add this to ensure libraries are visible to the loader
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_ROOT/lib

# Run Prediction for DenseNet
echo "Running DenseNet Predictions..."
python src/predict.py --model densenet --model_path runs/best_model_densenet.pt

# Run Prediction for MIL
echo "Running MIL Predictions..."
python src/predict.py --model mil --model_path runs/best_model_mil.pt