import numpy as np
from data_utils import *
from model_utils import *
import gpytorch
import torch
import pandas as pd
import argparse
import time

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parse file_number, loc, width, amp, and shape from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--file_number', type=int, default=0)
parser.add_argument('--loc', type=int, default=500)
parser.add_argument('--width', type=int, default=1)
parser.add_argument('--amp', type=float, default=-1.)
parser.add_argument('--shape', type=str, default='gaussian')
args = parser.parse_args()

# Load the data
data_dir = '../data/k2/'
filename = f'k2_{args.file_number}.fits'
x, y, y_err = load_k2_data(data_dir + filename)

# Scale data to be between 0 and 1
y = (y - np.min(y)) / (np.max(y) - np.min(y))
y_err = y_err / (np.max(y) - np.min(y))

# Inject anomalies
steps, y, anomaly_locs = inject_anomaly(
    y, 
    num_anomalies=1, 
    seed=args.file_number, 
    shapes=[args.shape],
    anomaly_stdev=args.width,
    anomaly_amp=args.amp,
    anomaly_idx=[args.loc]
)

# Standardize data to have mean 0 and std of 1
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y
y_err = y_err / std_y

# Create original copies of x and y
x_orig = np.copy(x)
y_orig = np.copy(y)
y_err_orig = np.copy(y_err)

# Convert to tensors
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
y_err_tensor = torch.tensor(y_err, dtype=torch.float32).to(device)

# Hyperparameters
which_metric = 'msll' # 'rmse', 'nlpd', msll, or default is 'mll'
num_anomalies = 3**len(anomaly_locs)
num_steps = len(x)
anomalous = np.zeros(num_steps) # 0 means non-anomalous, 1 means anomalous at that time step
initial_lengthscale = 0.5 # If None, no lengtshcale is used (default) and the theta parameter is the identity matrix
expansion_param = 2 # how many indices left and right to increase anomaly by

# Train GP model
model, likelihood, mll = train_gp(x_tensor, y_tensor, y_err_tensor, training_iterations=30, lengthscale=initial_lengthscale, device=device)
final_lengthscale = model.covar_module.base_kernel.rbf_kernel.lengthscale.item()

# Step 7 (repeat for every anomaly)
for i in range(num_anomalies):
    # Get subset of data that is flagged an non-anomalous
    x_sub = torch.tensor(x[anomalous == 0], dtype=torch.float32).to(device)
    y_sub = torch.tensor(y[anomalous == 0], dtype=torch.float32).to(device)
    y_err_sub = torch.tensor(y_err[anomalous == 0], dtype=torch.float32).to(device)

    # Re-fit the GP on non-anomalous data
    model, likelihood, mll = train_gp(x_sub, y_sub, y_err_sub, training_iterations=1, lengthscale=final_lengthscale, device=device)

    # Predict
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_tensor))
        pred_mean = observed_pred.mean.cpu().numpy()
        pred_var = observed_pred.variance.cpu().numpy()

    # Find index of largest deviation
    sig_dev = (pred_mean - y) / y_err
    index = np.argmax(sig_dev[anomalous == 0])
    print(f"\n\n New dip identified at anomalous index {index}, x[index] = {x[index]}")

    # Intialize variables for expanding anomalous region
    left_edge = index
    right_edge = index
    diff_metric = 1e6
    metric = 1e6
    
    # While the metric is decreasing, expand the anomalous edges
    while diff_metric > 0:
        # Subset x, y, and y_err
        subset = (((np.arange(len(x)) > right_edge) | (np.arange(len(x)) < left_edge)) & (anomalous == 0))
        x_sub = torch.tensor(x[subset], dtype=torch.float32).to(device)
        y_sub = torch.tensor(y[subset], dtype=torch.float32).to(device)
        y_err_sub = torch.tensor(y_err[subset], dtype=torch.float32).to(device)
        
        # Re-fit the GP on non-anomalous data
        model, likelihood, mll = train_gp(x_sub, y_sub, y_err_sub, training_iterations=1, lengthscale=final_lengthscale, device=device)

        # Predict
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Subset
            observed_pred_sub = likelihood(model(x_sub))
            pred_mean_sub = observed_pred.mean.cpu().numpy()
            
            # Full
            observed_pred_full = likelihood(model(x_tensor))
            pred_mean_full = observed_pred_full.mean.cpu().numpy()

        # Calculate metric difference
        old_metric = metric

        if which_metric == 'nlpd':
            metric = gpytorch.metrics.negative_log_predictive_density(observed_pred_sub, y_sub)
        elif which_metric == 'msll':
            metric = gpytorch.metrics.mean_standardized_log_loss(observed_pred_sub, y_sub)
        elif which_metric == 'rmse':
            metric = np.sqrt(np.mean((pred_mean_sub - y_sub.cpu().numpy())**2))
        else: # metric == mll
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = model(x_sub)
                metric = mll(output, y_sub)

        diff_metric = old_metric - metric # smaller is better
        print(f"Old metric: {old_metric} - New metric: {metric} = Diff metric: {diff_metric}")

        # Expand anomalous region on both sides
        if left_edge >= 0 + expansion_param:
            left_edge -= expansion_param
        if right_edge < len(x) - expansion_param:
            right_edge += expansion_param

        print(f"Anomaly index {i} x[i] {x[i]}, left_edge:right_edge {left_edge}:{right_edge}")

    # Update anomalous array and remove anomalies from y
    y[left_edge:right_edge] = pred_mean_full[left_edge:right_edge]
    anomalous[left_edge:right_edge] = 1

# Check whether every anomaly_locs was identified in the anomalous array
identified = np.zeros(len(anomaly_locs))
flagged_anomalies = np.where(anomalous == 1)

for i in range(len(anomaly_locs)):
    # Define anomaly_range as +/- 1 width of the anomaly
    anomaly_range = np.arange(int(anomaly_locs[i]) - args.width, int(anomaly_locs[i]) + args.width)

    # If at least one index in the anomaly_range is identified, set identified to 1
    if np.any(np.isin(anomaly_range, flagged_anomalies)):
        identified[i] = 1

identified_ratio = np.sum(identified) / len(anomaly_locs)

# Put results into a dictionary
column_names = ['filename', 'amp', 'width_stdev', 'shape', 'location_idx', 'flagged_anomalies', 'identified', 'identified_ratio']
results = {
    'filename': filename, 
    'amp': args.amp, 
    'width_stdev': args.width, 
    'shape': args.shape,
    'location_idx': args.loc, 
    'flagged_anomalies': flagged_anomalies, 
    'identified': identified, 
    'identified_ratio': identified_ratio
}

# Convert results to a dataframe
gp_results = pd.DataFrame(results, columns=column_names)

# Write gp_results to results_dir
results_dir = '../results/'
results_filename = 'gp_vary_size_100_repeats_3files.csv'

# Append results to results file if it exists, else create it
try:
    existing_results = pd.read_csv(results_dir + results_filename)
    gp_results = pd.concat([existing_results, gp_results], axis=0)
    gp_results.to_csv(results_dir + results_filename, index=False)
except FileNotFoundError:
    gp_results.to_csv(results_dir + results_filename, index=False)   

# Get running time
end_time = time.time()
run_time = end_time - start_time
print(f"Total runtime {run_time} \n---\n")
