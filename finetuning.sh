#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1

source $HOME/venv/bin/activate
python -u finetuning_sentence_transformer.py $@
