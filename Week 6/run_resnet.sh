#!/bin/bash
#SBATCH --job-name=ridge_all_layers_tuning          # job name
#SBATCH --output=ridge_pca_all_layers_stratified.log


#SBATCH --gres=gpu:1              # request 1 GPU, commented out
#SBATCH --mem=64G                 # memory
#SBATCH --cpus-per-task=4         # optional: useful for dataloading
#SBATCH --time=01:00:00           # 2-hour time limit

# Load your environment/modules
# module load python/3.9            # only if your system uses modules

# Activate your virtual environment (adjust to your setup)
source conda activate nx414N  # or conda activate myenv

# Run your Python script
python ridge_pca_all_layers.py
