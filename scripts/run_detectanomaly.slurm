#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=gpu # sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --gres=gpu:1
#SBATCH --array=17,18,27,28,44,3,9,29,43,48,70
#SBATCH --cpus-per-task=3 ## # cores per task
#SBATCH --mem=24gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

#SBATCH --job-name=whitenoise
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
widths=(0.001 0.01 0.0243 0.0591 0.1436 0.349 0.8483 2.062 5) 
depths=(5 4 3 2 1)
shape='gaussian'
results_filename='gp_periodic_vs_flat_new_detection'
detection_range=50
min_contiguous=10
file_number=$SLURM_ARRAY_TASK_ID # Each array job gets a single file_number based on the task ID
whitenoise=1 # if 1, test on whitenoise. Else, test on real k2 files

# Run for each width, amplitude, and shape
for width in "${widths[@]}"
do
    for depth in "${depths[@]}" 
    do
        # Reset the count
        count=1

        # Repeat 100 times
        while [ $count -le 100 ]
        do
            echo "Running iteration $count"
            
            # Each time, select a random loc
            loc=$((600 + RANDOM % (3200 - 600 + 1)))

            # Run the script with the selected file_number and anomaly parameters
            echo "Running file_number: $file_number with anomaly parameters loc: $loc, width_scale: $width, depth_scale: $depth, shape: $shape, whitenoise: $whitenoise, and save to $results_filename"
            python evaluate_gp_detectanomaly.py --file_number $file_number --loc $loc --width $width --depth $depth --shape $shape --results_filename $results_filename --detection_range $detection_range --min_contiguous $min_contiguous --whitenoise $whitenoise

            ((count++))
        done
    done
done
