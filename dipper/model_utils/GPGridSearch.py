import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from gp_model import train_gp
from utils import check_identified_anomalies
from scipy.signal import find_peaks, periodogram, windows
from scipy.ndimage import gaussian_filter1d
import gc

# Create a class that takes in a lightcurve (x, y, optionally y_err), and computes all possible start and end points of an anomaly.
# It then, for each start and end point (anomalous interval), trains a GP model on the non-anomalous data, and computes the log likelihood of the anomalous interval.
# The class returns the anomalous interval with the highest log likelihood.
# There should be a prior on the length of an anomaly, such as no more than 10% of the lightcurve and shortest is the nyquist frequency.

class GPGridSearch:
    def __init__(
            self,
            x,
            y,
            y_err=None,
            reset_method='mask', # 'mask', 'linear_interp', 'mean', or 'sinusoidal' # TODO: what should I replace y value with? open question
            which_metric='mll', # 'nlpd', 'msll', 'rmse', 'mll', or default is 'll' (log-likelihood)
            initial_lengthscale=None, # If None, no lengthscale is used (default) and the theta parameter is the identity matrix
        ):

        # Initialize
        self.x = x
        self.y = y
        self.y_err = y_err
        self.reset_method = reset_method
        self.which_metric = which_metric
        self.initial_lengthscale = initial_lengthscale
        self.num_steps = len(x)
        self.anomalous = np.zeros(self.num_steps) # 0 means non-anomalous, 1 means anomalous at that time step

        # Create original copies of x, y, and y_err
        self.x_orig = np.copy(x)
        self.y_orig = np.copy(y)
        self.y_err_orig = np.copy(y_err)

        # Store anomaly indicators (0 = non-anomalous, 1 = anomalous)
        self.anomalous = np.zeros(self.num_steps)

        # Set constraints for anomaly length
        min_anomaly_len = int(1 / (2 * np.median(np.diff(x))))  # Nyquist frequency
        max_anomaly_len = int(0.1 * self.num_steps)  # Max 10% of total steps

        # Possible anomaly intervals
        self.anomaly_intervals = [(i, j) for i in range(self.num_steps)
                                  for j in range(i + min_anomaly_len, min(i + max_anomaly_len, self.num_steps))]

    def find_anomalous_interval(self, device=torch.device("cpu"), training_iterations=10, filename=""):
        # Initialize
        max_metric = -np.inf
        best_interval = None

        if filename == "":
            save_to_txt = False
        else:
            # Create csv file to save results
            save_to_txt = True

            # write header
            with open(filename, 'w') as f:
                f.write('start,end,metric\n')


        # Iterate over each possible anomaly interval
        for start, end in self.anomaly_intervals:
            # Create data for training according to reset method
            if self.reset_method == 'mask':
                mask = np.ones(self.num_steps, dtype=bool)
                mask[start:end] = False
                x_train = torch.tensor(self.x[mask], dtype=torch.float32).to(device)
                y_train = torch.tensor(self.y[mask], dtype=torch.float32).to(device)
                y_err_train = torch.tensor(self.y_err[mask], dtype=torch.float32).to(device) if self.y_err is not None else None
            elif self.reset_method == 'linear_interp':
                x_train = torch.tensor(self.x, dtype=torch.float32).to(device)
                y_train = torch.tensor(self.y, dtype=torch.float32).to(device)
                y_err_train = torch.tensor(self.y_err, dtype=torch.float32).to(device) if self.y_err is not None else None
                y_train[start:end] = torch.tensor(np.interp(self.x[start:end], self.x, self.y), dtype=torch.float32).to(device)
                y_err_train[start:end] = torch.tensor(np.interp(self.x[start:end], self.x, self.y_err), dtype=torch.float32).to(device) if self.y_err is not None else None
            elif self.reset_method == 'mean':
                x_train = torch.tensor(self.x, dtype=torch.float32).to(device)
                y_train = torch.tensor(self.y, dtype=torch.float32).to(device)
                y_err_train = torch.tensor(self.y_err, dtype=torch.float32).to(device) if self.y_err is not None else None
                y_train[start:end] = torch.tensor(np.mean(self.y), dtype=torch.float32).to(device)
                y_err_train[start:end] = torch.tensor(np.mean(self.y_err), dtype=torch.float32).to(device) if self.y_err is not None else None
            # elif self.reset_method == 'sinusoidal':
            #     freqs, power = periodogram(self.y)
            #     peaks, _ = find_peaks(power)
            #     if len(peaks) == 0:
            #         print("No peaks found in power spectrum, using shoulder instead")
            #         smooth_power = gaussian_filter1d(power, 2)
            #         slope = np.gradient(smooth_power, freqs)
            #         shoulder_idx = np.where(slope < 0)[0][0]
            #         dominant_period = 1 / freqs[shoulder_idx]
                    
            #     else:
            #         dominant_peak = peaks[np.argmax(power[peaks])]
            #         dominant_period = 1 / freqs[dominant_peak]

                # Create sinusoid with period of peak in power spectrum
                # TODO: How do I know the offset and everything?

            # Train GP model on non-anomalous data
            model, likelihood, mll = train_gp(x_train, y_train, y_err_train, lengthscale=self.initial_lengthscale, training_iterations=training_iterations, device=device)

            # Evaluate metric for prediction on non-anomalous interval
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_pred = likelihood(model(x_train))
                f_pred = model(x_train)

            if self.which_metric == 'nlpd':
                metric = gpytorch.metrics.negative_log_predictive_density(y_pred, y_train)

            elif self.which_metric == 'msll':
                metric = gpytorch.metrics.mean_standardized_log_loss(y_pred, y_train)

            elif self.which_metric == 'rmse':
                pred_mean = y_pred.mean.cpu().numpy()
                metric = np.sqrt(np.mean((pred_mean - y_train.cpu().numpy())**2))

            elif self.which_metric == 'mll':
                metric = mll(f_pred, y_train)

            else: # Default to log-likelihood
                metric = y_pred.log_prob(y_train)

            # Update best interval
            if metric > max_metric:
                max_metric = metric
                best_interval = (start, end)

            print(f"Anomaly interval: {start}-{end}, Metric: {metric}")

            # Save results to csv if save_to_csv is True
            if save_to_txt:
                with open(filename, 'a') as f:
                    f.write(f'{start},{end},{metric}\n')

            # Delete all variables to free up memory
            del model
            del likelihood
            del mll
            del y_pred
            del f_pred
            del metric
            del mask
            del x_train
            del y_train
            del y_err_train
            torch.cuda.empty_cache()
            gc.collect()

        return best_interval, max_metric

    