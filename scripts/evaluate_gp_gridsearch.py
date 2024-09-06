# add ../src directory to path
import sys
sys.path.append('../src')

# imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import *
from GPGridSearch import GPGridSearch
import gpytorch
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_number', type=int, default=0)
args = parser.parse_args()

# Data and anomaly parameters
shape = "gaussian"
width_scale = 1.6
depth_scale = 0.5
loc = 2000

# Load the data
data_dir = '../data/k2/'
data_filename = f'k2_{args.file_number}.fits'
x, y, y_err = load_k2_data(data_dir + data_filename)

# Scale data to be between 0 and 1
y = (y - np.min(y)) / (np.max(y) - np.min(y))
y_err = y_err / (np.max(y) - np.min(y))

# Inject anomalies
steps, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(
    y, 
    num_anomalies=1, 
    seed=args.file_number, 
    shapes=[shape],
    width_scale=width_scale,
    depth_scale=depth_scale,
    anomaly_idx=[loc]
)

# Standardize data to have mean 0 and std of 1
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y
y_err = y_err / std_y

# Print anomaly_locs, both indices and x[i]
print("Anomaly locations:")
print(anomaly_locs)
for i in anomaly_locs: print(x[int(i)])

# Hyperparameters for grid search
which_metric = 'mll'
initial_lengthscale = 0.5
training_iterations = 30
plot = False
filename  = f'../results/grid_search/gp_initial_{args.file_number}.txt'

gp_detector = GPGridSearch(
    x,
    y,
    y_err,
    which_metric=which_metric,
    initial_lengthscale=initial_lengthscale
)

best_interval, max_metric = gp_detector.find_anomalous_interval(device=device, training_iterations=training_iterations, filename=filename)
print(f"Best interval: {best_interval}, Max metric: {max_metric}, saved to {filename}")