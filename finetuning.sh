#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1

source $HOME/venv/bin/activate
python -u finetuning_sentence_transformer.py $@
