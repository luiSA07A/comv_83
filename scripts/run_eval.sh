#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --output=logs/eval_%j.out
#SBATCH --partition=CPUQ
#SBATCH --time=00:10:00

source ~/venvs/odelia/bin/activate

echo "========================================"
echo "EVALUATING DENSENET"
echo "========================================"
python src/evaluate_results.py --json runs/preds_densenet.json --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025

echo -e "\n========================================"
echo "EVALUATING MIL"
echo "========================================"
python src/evaluate_results.py --json runs/preds_mil.json --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025