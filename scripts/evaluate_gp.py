# add ../src directory to path
import sys
sys.path.append('../src')

# imports
import numpy as np
from utils import *
from GPDetectAnomaly import GPDetectAnomaly
import gpytorch
import torch
import pandas as pd
import argparse
import time

# Start time
start_time = time.time()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parse file_number, loc, width, amp, and shape from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--file_number', type=int, default=0)
parser.add_argument('--loc', type=int, default=500)
parser.add_argument('--width', type=float, default=0.1)
parser.add_argument('--depth', type=float, default=-1.)
parser.add_argument('--shape', type=str, default='gaussian')
parser.add_argument('--results_filename', type=str, default='gp_results')
args = parser.parse_args()

# Load the data
data_dir = '../data/k2/'
filename = f'k2_{args.file_number}.fits'
x, y, y_err = load_k2_data(data_dir + filename)

# Scale data to be between 0 and 1
y = (y - np.min(y)) / (np.max(y) - np.min(y))
y_err = y_err / (np.max(y) - np.min(y))

# Inject anomalies
steps, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(
    y, 
    num_anomalies=1, 
    seed=args.file_number, 
    shapes=[args.shape],
    width_scale=args.width,
    depth_scale=args.depth,
    anomaly_idx=[args.loc]
)

# Standardize data to have mean 0 and std of 1
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y
y_err = y_err / std_y

# Hyperparameters
which_metric = 'msll' # 'rmse', 'nlpd', msll, or default is 'mll'
num_anomalies = 3**len(anomaly_locs)
initial_lengthscale = 0.5 # If None, no lengtshcale is used (default) and the theta parameter is the identity matrix
expansion_param = 2 # how many indices left and right to increase anomaly by
training_iterations = 30
plot = False

gp_detector = GPDetectAnomaly(
    x,
    y,
    y_err,
    which_metric=which_metric,
    num_anomalies=num_anomalies,
    initial_lengthscale=initial_lengthscale,
    expansion_param=expansion_param,
)

x, y, anomalous = gp_detector.detect_anomaly(
    training_iterations=training_iterations, 
    plot=plot,
    device=device,
    anomaly_locs=anomaly_locs,
    anomaly_fwhm=anomaly_fwhm
)

# Check identified anomalies
flagged_anomalies = np.where(anomalous == 1)
identified, identified_ratio = check_identified_anomalies(anomaly_locs, flagged_anomalies, anomaly_fwhm)

# Put results into a dictionary
column_names = ['filename', 'depth_scale', 'width_scale', 'shape', 'anomaly_amp', 'anomaly_fwhm', 
                'location_idx', 'flagged_anomalies', 'identified', 'identified_ratio']
results = {
    'filename': filename, 
    'depth_scale': args.depth, 
    'width_scale': args.width, 
    'shape': args.shape,
    'location_idx': args.loc, 
    'anomaly_amp': anomaly_amp,
    'anomaly_fwhm': anomaly_fwhm,
    'flagged_anomalies': flagged_anomalies, 
    'identified': identified, 
    'identified_ratio': identified_ratio
}

# Convert results to a dataframe
gp_results = pd.DataFrame(results, columns=column_names)

# Write gp_results to results_dir
results_dir = '../results/'

# Append results to results file if it exists, else create it
try:
    existing_results = pd.read_csv(results_dir + args.results_filename + '.csv')
    gp_results = pd.concat([existing_results, gp_results], axis=0)
    gp_results.to_csv(results_dir + args.results_filename + '.csv', index=False)

except FileNotFoundError:
    gp_results.to_csv(results_dir + args.results_filename + '.csv', index=False)   

# Get running time
end_time = time.time()
run_time = end_time - start_time
print(f"Total runtime {run_time} \n---\n")
