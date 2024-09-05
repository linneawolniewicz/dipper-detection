import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from gp_model import train_gp
from utils import check_identified_anomalies

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
            which_metric='mll', # 'nlpd', 'msll', 'rmse', 'mll', or default is 'll' (log-likelihood)
            initial_lengthscale=None, # If None, no lengthscale is used (default) and the theta parameter is the identity matrix
        ):

        # Initialize
        self.x = x
        self.y = y
        self.y_err = y_err
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

    def find_anomalous_interval(self, device=torch.device("cpu"), training_iterations=10):
        # Initialize
        max_metric = -np.inf
        best_interval = None

        # Iterate over each possible anomaly interval
        for start, end in self.anomaly_intervals:
            # Select non-anomalous data for training
            mask = np.ones(self.num_steps, dtype=bool)
            mask[start:end] = False
            x_train = torch.tensor(self.x[mask], dtype=torch.float32).to(device)
            y_train = torch.tensor(self.y[mask], dtype=torch.float32).to(device)
            y_err_train = torch.tensor(self.y_err[mask], dtype=torch.float32).to(device) if self.y_err is not None else None

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

        return best_interval, max_metric

    