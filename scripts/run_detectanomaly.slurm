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

#SBATCH --job-name=gp_78lc
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
widths=$(awk 'BEGIN {for(i=0; i<10; i++) print 10^((log(0.1)/log(10)) + i*((log(5)/log(10) - log(0.1)/log(10))/9))}') # 10 values from 0.1 to 5 in log space 
depths=(1 2 3 4 5)
shapes=('gaussian')
results_filename='gp_vary_size_100_snr'
# file_numbers=(3 18 48)

# Run for each width, amplitude, and shape
for width in $widths
do
    for depth in "${depths[@]}" 
    do
        for shape in "${shapes[@]}" 
        do
        # for file_number in "${file_numbers[@]}"
        # do
            # Reset the count
            count=1

            # Repeat 100 times
            while [ $count -le 100 ]
            do
                echo "Running iteration $count"
                
                # Each time, select a random loc
                loc=$((600 + RANDOM % (3200 - 600 + 1)))

                # Each time, select a random file_number from [0, 77]
                file_number=$((RANDOM % 78))

                # Run the script with the selected file_number and anomaly parameters
                echo "Running file_number: $file_number with anomaly parameters loc: $loc, width_scale: $width, depth_scale: $depth, shape: $shape, and save to $results_filename"
                python evaluate_gp_detectanomaly.py --file_number $file_number --loc $loc --width $width --depth $depth --shape $shape --results_filename $results_filename

                ((count++))
            # done
            done
        done
    done
done
