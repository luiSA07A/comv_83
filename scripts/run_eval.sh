#!/bin/bash
#SBATCH --job-name=odelia_eval
#SBATCH --partition=GPUQ
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/eval_%j.out

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/odelia/bin/activate

python src/evaluate_results.py