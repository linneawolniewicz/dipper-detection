# imports
import numpy as np
from dipper.model_utils.utils import *
from dipper.data_utils.utils import *
from dipper.model_utils.GPGrowDeviant import GPGrowDeviant
from astropy.io import fits
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

# Parse file_number, loc, period_scale, amp, and shape from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--file_number', type=int, default=0)
parser.add_argument('--loc', type=int, default=500)
parser.add_argument('--period_scale', type=float, default=0.1)
parser.add_argument('--snr', type=float, default=-1.)
parser.add_argument('--shape', type=str, default='gaussian')
parser.add_argument('--min_contiguous', type=int, default=1)
parser.add_argument('--detection_range', type=int, default=100)
parser.add_argument('--results_filename', type=str, default='gp_results')
parser.add_argument('--whitenoise', type=int, default=0)
parser.add_argument('--len_deviant', type=int, default=0)
args = parser.parse_args()

# Load the data
data_dir = '../data/k2/'  # TODO: Update path
filename = f'k2_{args.file_number}.fits'

# If whitenoise is 1, load x, y, and y_err from '../data/whitenoise.fits'
if args.whitenoise == 1:
    hdu = fits.open('../data/whitenoise.fits')  # TODO: Update path
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

# Inject anomalies
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
which_metric = 'mll' # 'rmse', 'nlpd', msll, or default is 'mll'
num_anomalies = 3**len(anomaly_locs)
initial_lengthscale = 0.7 # If None, no lengthscale is used (default) and the theta parameter is the identity matrix
expansion_param = 3 # how many indices left and right to increase anomaly by
training_iterations = 50
plot = False

gp_detector = GPGrowDeviant(
    x,
    y,
    y_err,
    which_metric=which_metric,
    num_anomalies=num_anomalies,
    initial_lengthscale=initial_lengthscale,
    expansion_param=expansion_param,
    len_deviant=args.len_deviant
)

gp_detector.detect_anomaly(
    training_iterations=training_iterations, 
    plot=plot,
    device=device,
    anomaly_locs=anomaly_locs,
    detection_range=args.detection_range
)

# Check identified anomalies
identified, identified_ratio = check_identified_anomalies(
    anomaly_locs, 
    gp_detector.anomalous, 
    args.detection_range,
    args.min_contiguous
)

print(f"Identified {identified} out of {len(anomaly_locs)} anomalies")
flagged_anomalies = np.where(gp_detector.anomalous == 1)

# Put results into a dictionary
column_names = [
    'filename', 
    'snr', 
    'period_scale', 
    'shape', 
    'location_idx', 
    'anomaly_amp', 
    'anomaly_fwhm', 
    'flagged_anomalies', 
    'detection_range', 
    'min_contiguous', 
    'identified', 
    'identified_ratio',
    'len_deviant'
]

results = {
    'filename': filename, 
    'snr': args.snr, 
    'period_scale': args.period_scale, 
    'shape': args.shape,
    'location_idx': args.loc, 
    'anomaly_amp': anomaly_amp,
    'anomaly_fwhm': anomaly_fwhm,
    'flagged_anomalies': str(flagged_anomalies), 
    'detection_range': args.detection_range,
    'min_contiguous': args.min_contiguous,
    'identified': str(identified), 
    'identified_ratio': identified_ratio,
    'len_deviant': args.len_deviant
}

# Convert results to a dataframe
gp_results = pd.DataFrame([results], columns=column_names)

# Write gp_results to results_dir
results_dir = '../results/'  # TODO: Update path
if args.whitenoise == 1:
    final_filename = results_dir + args.results_filename + '/whitenoise.csv'
else:
    final_filename = results_dir + args.results_filename + '/lc_' + str(args.file_number) + '.csv'
print(f"Writing results to {final_filename}")

# Append results to results file if it exists, else create it
try:
    existing_results = pd.read_csv(final_filename)
    gp_results = pd.concat([existing_results, gp_results], axis=0)
    gp_results.to_csv(final_filename, index=False)

except FileNotFoundError:
    gp_results.to_csv(final_filename, index=False)   

# Get running time
end_time = time.time()
run_time = end_time - start_time
print(f"Total runtime {run_time} \n---\n")
