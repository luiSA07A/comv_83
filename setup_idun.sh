#!/bin/bash
# ============================================================
#  One-time setup script — run this on IDUN login node
#  before submitting any jobs.
#  Usage:  bash setup_idun.sh
# ============================================================

echo "=== Setting up ODELIA project environment on IDUN ==="

# Load base modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Create virtual environment in home dir
VENV_PATH=~/venvs/odelia
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH ..."
    python -m venv $VENV_PATH
else
    echo "Virtual environment already exists."
fi

source $VENV_PATH/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install monai[all] nibabel numpy pandas scikit-learn matplotlib seaborn tqdm

echo ""
echo "=== Setup complete! ==="
echo "To activate: source ~/venvs/odelia/bin/activate"
echo "To submit DenseNet job: sbatch scripts/train_densenet.sh"
echo "To submit MIL job:      sbatch scripts/train_mil.sh"
echo "To check queue:         squeue -u \$USER"
echo "To cancel a job:        scancel <JOB_ID>"
