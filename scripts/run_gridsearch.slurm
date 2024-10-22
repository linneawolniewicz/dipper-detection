#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=gpu # sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --array=3,9,17,18,27,28,29,43,44,48,70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3 ## # cores per task
#SBATCH --mem=8gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

#SBATCH --job-name=gp_grid
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

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
period_scales=(0.001 0.003 0.011 0.039 0.13 0.439 1.484 5.000)
snrs=(1 2 3 4)
shape='gaussian'
save_dir='grid_search_multiple_loc_2'
window_slide_step=20
window_size_step=60
assume_independent=1 # 1=True, 0=False
which_metric='mll'
min_contiguous=10
detection_range=50
file_number=$SLURM_ARRAY_TASK_ID # Each array job gets a single file_number based on the task ID
whitenoise=0 # if 1, test on whitenoise. Else, test on real k2 files
num_repeats=10

for period_scale in "${period_scales[@]}"
do
    for snr in "${snrs[@]}" 
    do
        # Reset the count
        count=1

        # Repeat num_repeats times
        while [ $count -le $num_repeats ]
        do
            echo "Running iteration $count"
            
            # Each time, select a random loc
            loc=$((600 + RANDOM % (3200 - 600 + 1)))
            echo "Running grid search on file $file_number with anomaly parameters loc: $loc, period_scale: $period_scale, and snr: $snr"
            python evaluate_gp_gridsearch.py --file_number $file_number --save_dir $save_dir --period_scale $period_scale --snr $snr --loc $loc --shape $shape --window_slide_step $window_slide_step --window_size_step $window_size_step --assume_independent $assume_independent --which_metric $which_metric --min_contiguous $min_contiguous --detection_range $detection_range --whitenoise $whitenoise
        
            ((count++))
        done
    done
done

# NOTE: this can only run with large amounts of memory