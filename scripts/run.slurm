#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=sadow # sadow, gpu, shared, kill-shared, exclusive-long
#SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6 ## # cores per task
#SBATCH --mem=48gb ## max amount of memory per node you require
#SBATCH --time=14-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

#SBATCH --job-name=gp
#SBATCH --output=job.out
#SBATCH --output=../logs/slurm_output/job-%A.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
source activate dipper_detect

# use this command to run a python script
# python evaluate_gp.py 

# use this command to run an ipynb and save outputs in the notebook
# jupyter nbconvert --execute --clear-output file.ipynb 

# Another command to create a .py script, then run that from a ipynb:
# jupyter nbconvert file.ipynb --to python
# python file.py

# Create a for loop to run the script with different arguments
file_numbers=(3 18 48)
locs=(500 1750 3000)
widths=(2 14 26 38 50)
amps=(-5.0 -4.0 -3.0 -2.0 -1.0)
shapes=('gaussian')

for file_number in "${file_numbers[@]}" 
do
    for loc in "${locs[@]}" 
    do
        for width in "${widths[@]}" 
        do
            for amp in "${amps[@]}" 
            do
                for shape in "${shapes[@]}" 
                do
                    echo "Running file_number: $file_number with anomaly parameters loc: $loc, width: $width, amp: $amp, shape: $shape"
                    python evaluate_gp.py --file_number $file_number --loc $loc --width $width --amp $amp --shape $shape
                done
            done
        done
    done
done
