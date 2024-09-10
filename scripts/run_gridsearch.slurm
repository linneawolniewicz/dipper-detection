#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=gpu # sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --array=3,18,48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6 ## # cores per task
#SBATCH --mem=192gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

#SBATCH --job-name=gp_grid_search
#SBATCH --output=job.out
#SBATCH --output=../logs/slurm_output/job-%A.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile and activate environment
source ~/profiles/auto.profile
source activate dipper_detect

# use this command to run a python script
# python evaluate_gp.py 

# use this command to run an ipynb and save outputs in the notebook
# jupyter nbconvert --execute --clear-output file.ipynb 

# Another command to create a .py script, then run that from a ipynb:
# jupyter nbconvert file.ipynb --to python
# python file.py

save_dir='grid_search_snr5_period01'

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo "Running file_number: $SLURM_ARRAY_TASK_ID"
python evaluate_gp_gridsearch.py --file_number $SLURM_ARRAY_TASK_ID --save_dir $save_dir

# NOTE: this can only run with 216 gb.