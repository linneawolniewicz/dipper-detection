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
import pandas as pd

# Start time
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_number', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="grid_search")
parser.add_argument('--period_scale', type=float, default=0.2)
parser.add_argument('--snr', type=float, default=2)
parser.add_argument('--loc', type=int, default=500)
parser.add_argument('--shape', type=str, default="gaussian")
parser.add_argument('--min_contiguous', type=int, default=1)
parser.add_argument('--detection_range', type=int, default=100)
parser.add_argument('--window_slide_step', type=int, default=10)
parser.add_argument('--window_size_step', type=int, default=50)
parser.add_argument('--assume_independent', type=int, default=1)
parser.add_argument('--which_metric', type=str, default="mll")
parser.add_argument('--whitenoise', type=int, default=0)
args = parser.parse_args()

assume_independent = True if args.assume_independent == 1 else False

# Load the data
data_dir = '../data/k2/'
filename = f'k2_{args.file_number}.fits'

# If whitenoise is 1, load x, y, and y_err from '../data/whitenoise.fits'
if args.whitenoise == 1:
    hdu = fits.open('../data/whitenoise.fits')
    x = hdu[0].data[0]
    y = hdu[0].data[1]
    y_err = hdu[0].data[2]

    # Convert to native byte order
    x = x.byteswap().newbyteorder()
    y = y.byteswap().newbyteorder()
    y_err = y_err.byteswap().newbyteorder()

else:
    x, y, y_err = load_k2_data(data_dir + filename)

# Scale data to be between 0 and 1
y = (y - np.min(y)) / (np.max(y) - np.min(y))
y_err = y_err / (np.max(y) - np.min(y))

# Inject anomalies at a random location
steps, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(
    y, 
    num_anomalies=1, 
    seed=args.file_number, 
    shapes=[args.shape],
    period_scale=args.period_scale,
    snr=args.snr,
    anomaly_idx=[args.loc]
)

# Standardize data to have mean 0 and std of 1
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y
y_err = y_err / std_y

# Hyperparameters
min_anomaly_len = max(1, int(1 / (2 * np.median(np.diff(x)))))  # Nyquist frequency with minimum of 1
max_anomaly_len = int(0.1 * len(x))  # Max 10% of total time steps
initial_lengthscale = 0.5
training_iterations = 50

gp_detector = GPGridSearch(
    x,
    y,
    y_err,
    min_anomaly_len=min_anomaly_len,
    max_anomaly_len=max_anomaly_len,
    window_slide_step=args.window_slide_step,
    window_size_step=args.window_size_step,
    assume_independent=assume_independent,
    which_metric=args.which_metric,
    initial_lengthscale=initial_lengthscale,
)

gp_detector.find_anomalous_interval(
    device=device, 
    training_iterations=training_iterations,
    silent=True
)

# Find the 3 lowest mean_metric values and their related intervals
sorted_intervals = sorted(zip(gp_detector.mean_metrics, gp_detector.intervals), key=lambda x: x[0])
lowest_3_intervals = sorted_intervals[:3]

# Check identified anomalies
anomalous = np.zeros_like(y)

# Every interval in lowest_3_intervals is anomalous
for interval in lowest_3_intervals:
    start, end = interval[1]
    anomalous[start:end] = 1

identified, identified_ratio = check_identified_anomalies(
    anomaly_locs, 
    anomalous, 
    args.detection_range,
    args.min_contiguous
)

print(f"Number of anomaly intervals: {len(gp_detector.intervals)}, True anomaly idx: {args.loc}, Best interval: {gp_detector.best_interval}, Min metric: {gp_detector.min_metric}, identified: {identified}")

# Put results into a dictionary
column_names = [
    'file_number', 
    'snr', 
    'period_scale', 
    'shape', 
    'location_idx', 
    'anomaly_amp', 
    'anomaly_fwhm',
    'detection_range', 
    'min_contiguous', 
    'identified', 
    'identified_ratio',
    'window_slide_step',
    'window_size_step',
    'assume_independent',
    'which_metric',
    'num_intervals',
    'start',
    'end',
    'min_metric'
]

results = {
    'file_number': args.file_number, 
    'snr': args.snr, 
    'period_scale': args.period_scale, 
    'shape': args.shape,
    'location_idx': args.loc, 
    'anomaly_amp': anomaly_amp,
    'anomaly_fwhm': anomaly_fwhm,
    'detection_range': args.detection_range,
    'min_contiguous': args.min_contiguous,
    'identified': str(identified), 
    'identified_ratio': identified_ratio,
    'window_slide_step': args.window_slide_step,
    'window_size_step': args.window_size_step,
    'assume_independent': args.assume_independent,
    'which_metric': args.which_metric,
    'num_intervals': len(gp_detector.intervals),
    'start': gp_detector.best_interval[0],
    'end': gp_detector.best_interval[1],
    'min_metric': gp_detector.min_metric
}

# Convert results to a dataframe
gp_results = pd.DataFrame([results], columns=column_names)

# Write gp_results to results_dir
results_dir = '../results/'
if args.whitenoise == 1:
    final_filename = results_dir + args.save_dir + '/whitenoise.csv'
else:
    final_filename = results_dir + args.save_dir + '/lc_' + str(args.file_number) + '.csv'
print(f"Writing results to {final_filename}")

# Append results to results file if it exists, else create it
try:
    existing_results = pd.read_csv(final_filename)
    gp_results = pd.concat([existing_results, gp_results], axis=0)
    gp_results.to_csv(final_filename, index=False)

except FileNotFoundError:
    # Make directory if it doesn't exist
    gp_results.to_csv(final_filename, index=False)   

# Get running time
end_time = time.time()
run_time = end_time - start_time
print(f"Total runtime {run_time} \n---\n")