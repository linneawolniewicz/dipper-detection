#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=gpu # sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6 ## # cores per task
#SBATCH --mem=48gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

#SBATCH --job-name=gp_100
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

# Create a for loop to run the script with different arguments
widths=(2 4 8 16 24 32 48 64 80 96 112 128 144 160)
amps=(-1.5 -1.25 -1.0 -0.75)
shapes=('gaussian')

# Run for each width, amplitude, and shape
for width in "${widths[@]}" 
do
    for amp in "${amps[@]}" 
    do
        for shape in "${shapes[@]}" 
        do
            # Reset the count
            count=1

            # Repeat 100 times
            while [ $count -le 100 ]
            do
                echo "Running iteration $count"
                
                # Each time, select a random loc and file_number
                file_number=$((1 + RANDOM % 78))
                loc=$((600 + RANDOM % (3200 - 600 + 1)))

                # Run the script with the selected file_number and anomaly parameters
                echo "Running file_number: $file_number with anomaly parameters loc: $loc, width: $width, amp: $amp, shape: $shape"
                python evaluate_gp_tmp.py --file_number $file_number --loc $loc --width $width --amp $amp --shape $shape

                ((count++))
            done
        done
    done
done
