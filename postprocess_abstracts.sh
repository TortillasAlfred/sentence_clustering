#!/bin/bash                                      
#SBATCH --account=rpp-corbeilj                             # Account with resources
#SBATCH --cpus-per-task=32                                     # Number of CPUs
#SBATCH --mem=125G                                              # memory (per node)
#SBATCH --time=0-20:00                                        # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca               # Where to email
#SBATCH --mail-type=FAIL                                      # Email when a job fails

source ~/venvs/default/bin/activate

date
SECONDS=0

# You can access the array ID via $SLURM_ARRAY_TASK_ID

# The $@ transfers all args passed to this bash file to the Python script
# i.e. a call to 'sbatch $sbatch_args this_launcher.sh --arg1=0 --arg2=True'
# will call 'python my_script.py --arg1=0 --arg2=True'
python -u postprocess_abstracts.py --mesh_texts_dir=/scratch/magod/mobility_abstracts/mesh_texts/ --processed_dir=/scratch/magod/mobility_abstracts/processed/
# Utility to show job duration in output file
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
