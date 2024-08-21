#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=sadow # sadow, gpu, shared, kill-shared, exclusive-long
#SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

##BATCH --gres=gpu:1 ## if you need a gpu
#SBATCH --cpus-per-task=3 ## how many cores
#SBATCH --mem=16gb ## max amount of memory per node you require
#SBATCH --time=14-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

#SBATCH --job-name=wget_k2_lc
##SBATCH --output=logs/job-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

## Run wget .sh file 
./k2_lc.sh