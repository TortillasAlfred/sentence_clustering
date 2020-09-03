#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1

source $HOME/venv/bin/activate
python -u finetuning_sentence_transformer.py
