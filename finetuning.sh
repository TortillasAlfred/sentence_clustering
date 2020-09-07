#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1

source $HOME/venv/bin/activate
python -u finetuning_sentence_transformer.py $@
