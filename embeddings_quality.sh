#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1

source $HOME/venv/bin/activate
python -u embeddings_quality.py
