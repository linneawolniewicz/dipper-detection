# imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from dipper.model_utils.utils import *
from dipper.data_utils.utils import *
from dipper.model_utils.GPGridSearch import GPGridSearch
import gpytorch
import torch
import argparse
import time
import os

# Start time
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_number', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="grid_search")
parser.add_argument
args = parser.parse_args()

# Data and anomaly parameters
shape = "gaussian"
period_scale = 0.2
snr = 2

# Load the data
data_dir = '../data/k2/'
data_filename = f'k2_{args.file_number}.fits'
x, y, y_err = load_k2_data(data_dir + data_filename)

# Scale data to be between 0 and 1
y = (y - np.min(y)) / (np.max(y) - np.min(y))
y_err = y_err / (np.max(y) - np.min(y))

# Inject anomalies at a random location
steps, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(
    y, 
    num_anomalies=1, 
    seed=args.file_number, 
    shapes=[shape],
    period_scale=period_scale,
    snr=snr
)

# Standardize data to have mean 0 and std of 1
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y
y_err = y_err / std_y

# Print anomaly_locs, both indices and x[i]
print("\nFile number:", args.file_number)
print("Anomaly locations:")
print(anomaly_locs)
for i in anomaly_locs: print(x[int(i)])
print(f"Anomaly period_scale {period_scale}, snr {snr}, fwhm {anomaly_fwhm}, and amp {anomaly_amp}\n")

# Hyperparameters for grid search
which_metric = 'mll'
initial_lengthscale = 0.5
training_iterations = 50
plot = False
results_dir = "../results/"
save_file = results_dir + args.save_dir + f"/file_number_{args.file_number}.txt"

# If results_dir + args.save_dir doesn't exist, create it
if not os.path.exists(results_dir + args.save_dir + "/"):
    os.makedirs(results_dir + args.save_dir + "/")

gp_detector = GPGridSearch(
    x,
    y,
    y_err,
    which_metric=which_metric,
    initial_lengthscale=initial_lengthscale
)

best_interval, max_metric = gp_detector.find_anomalous_interval(
    device=device, 
    training_iterations=training_iterations, 
    filename=save_file
)
print(f"Best interval: {best_interval}, Max metric: {max_metric}, saved to {save_file}")

# Get running time
end_time = time.time()
run_time = end_time - start_time
print(f"Total runtime {run_time} \n---\n")