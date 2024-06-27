import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from data_utils import *
from model_utils import *
import gpytorch
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create an empty results dataframe of length 78 with column names
# ['filename', 'injected_anomalies', 'identified', 'identified_ratio', 'flagged_anomalies']
gp_results = pd.DataFrame(columns=['filename', 'injected_anomalies', 'identified', 'identified_ratio', 'flagged_anomalies'])

data_dir = '../data/k2/'
results_dir = '../results/'

# For each k2 fits file, laod the data, inject anomalies, detect them with a gp, and write to the results_dir
for i in range(1, 79):
    # Load a file and inject an anomaly
    filename = f'k2_{i}.fits'
    fits_file = fits.open(data_dir + filename)

    # Load data
    data = fits_file[1].data
    time = np.array(data['TIME'])
    pdc_flux = np.array(data['PDCSAP_FLUX'])
    pdc_err = np.array(data['PDCSAP_FLUX_ERR'])

    # Set x, y, and error
    x = time
    y = pdc_flux / np.nanmedian(pdc_flux)
    y_err = pdc_err / np.nanmedian(pdc_flux)

    # Clean
    clean = (y_err > 0.)
    x = x[clean]
    y = y[clean]
    y_err = np.abs(y_err[clean]) 

    # Shift to start at time 0
    x = x - np.min(x) 

    # Bin to 30-minute cadence
    num_bins = int(np.floor((np.max(x) - np.min(x)) * 48) + 1) # 48 bins per day
    x_bins = np.min(x) + np.arange(num_bins + 1) / 48.
    num_binned, bin_edges = np.histogram(x, bins= x_bins)
    num_binned = np.array(num_binned)
    y_binned, bin_edges = np.histogram(x, bins = x_bins, weights = y)
    var_binned, bin_edges = np.histogram(x, bins = x_bins, weights= 1 / y_err**2)
    y_err_binned = 1 / np.sqrt(np.array(var_binned))
    y_binned = np.array(y_binned)
    y_binned = y_binned / (num_binned + 0.001)
    x_binned = x_bins[0:num_bins] + (x_bins[1] - x_bins[0]) / 2. 
    x = x_binned
    y = y_binned
    y_err = y_err_binned

    # Clean
    clean = ((y > 0.) & ~np.isnan(y_err))
    x = x[clean]
    y = y[clean]
    y_err = y_err[clean]

    assert len(x) == len(y) == len(y_err), 'Lengths of x, y, and y_err must be the same'

    # Inject anomalies
    steps, y, anomaly_locs = inject_anomaly(y, num_anomalies=np.random.randint(1, 3), seed=i)

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
    num_anomalies = 2**len(anomaly_locs)
    num_steps = len(x)
    anomalous = np.zeros(num_steps) # 0 means non-anomalous, 1 means anomalous at that time step
    initial_lengthscale = None #0.3**2
    expansion_param = 2 # how many indices left and right to increase anomaly by

    # Train GP model
    model, likelihood, mll = train_gp(x_tensor, y_tensor, y_err_tensor, training_iterations=30, lengthscale=initial_lengthscale, device=device)
    final_lengthscale = model.covar_module.base_kernel.lengthscale.item()

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

            # Predict on subset
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(x_sub))
                pred_mean = observed_pred.mean.cpu().numpy()
                pred_var = observed_pred.variance.cpu().numpy()

            # Calculate metric difference
            old_metric = metric

            if which_metric == 'nlpd':
                metric = gpytorch.metrics.negative_log_predictive_density(observed_pred, y_sub)
            elif which_metric == 'msll':
                metric = gpytorch.metrics.mean_standardized_log_loss(observed_pred, y_sub)
            elif which_metric == 'rmse':
                metric = np.sqrt(np.mean((pred_mean - y_sub.cpu().numpy())**2))
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

        # Update anomalous array
        y[left_edge:right_edge] = pred_mean[left_edge:right_edge]
        anomalous[left_edge:right_edge] = 1

    # Check whether every anomaly_locs was identified in the anomalous array
    identified = np.zeros(len(anomaly_locs))
    flagged_anomalies = np.where(anomalous == 1)

    for i in range(len(anomaly_locs)):
        anomaly = int(anomaly_locs[i])

        if np.isin(anomaly, flagged_anomalies):
            identified[i] = 1
        else:
            print(f"Anomaly at index {anomaly} was not identified in the anomalous array")

    identified_ratio = np.sum(identified) / len(anomaly_locs)

    print(f"Injected anomaly centers: {anomaly_locs}")
    print(f"Anomalies identified: {identified}")
    print(f"Ratio of anomalies identified: {identified_ratio}")
    print(f"Flagged anomalies: {flagged_anomalies}")

    # Append results to results dataframe
    new_results = {'filename': filename, 'injected_anomalies': anomaly_locs, 'identified': identified, 'identified_ratio': identified_ratio, 'flagged_anomalies': flagged_anomalies}
    gp_results = gp_results.append(new_results, ignore_index=True)  

# Write gp_results to results_dir
gp_results.to_csv(results_dir + 'gp_results.csv')        
