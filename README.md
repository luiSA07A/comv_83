# ODELIA 2025: 3D DCE-MRI Breast Cancer Classification (Group 4)

This repository contains the pipeline for classifying 3D DCE-MRI volumes into Normal, Benign, and Malignant categories using MONAI and PyTorch.

## 🚀 Architectural Highlights
- **Unilateral Processing:** Each breast is treated as an independent study to maximize information density.
- **Cost-Sensitive Learning:** Implemented weighted Cross-Entropy Loss to prioritize clinical sensitivity (85.7%).
- **Sustainability:** Automated energy tracking with Tesla-equivalent kilometer reporting.

## 🛠️ Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd comv_83

# Setup Environment
conda create -n odelia python=3.11
conda activate odelia
pip install torch monai scikit-learn pandas
