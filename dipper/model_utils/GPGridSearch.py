import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from .gp_model import train_gp # TODO: Figure out how to do this import
from .utils import check_identified_anomalies
from scipy.signal import find_peaks, periodogram, windows
from scipy.ndimage import gaussian_filter1d
import gc

class GPGridSearch:
    def __init__(
            self,
            x,
            y,
            y_err=None,
            min_anomaly_len = 1,
            max_anomaly_len = 400,
            window_slide_step=1,
            window_size_step=1,
            assume_independent=True, 
            which_metric='mll', 
            initial_lengthscale=None, 
        ):
        """
        Method that searched over every possible interval that could be anomalous, and returns the one that is most likely to be anomalous.
        Assumes independence between points for speed, and takes shortcuts to reduce computation.
        Fits GP to data without the interval, and evaluates the log-likelihood of the data with the interval. Anomalous interval is the one with the lowest log-likelihood.
        There is a prior on the length of an anomaly, such as no more than 10% of the lightcurve and shortest is the nyquist frequency.

        Args:
            x (np.ndarray): Time values
            y (np.ndarray): Flux values
            y_err (np.ndarray): Flux error values
            min_anomaly_len (int): Minimum length of an anomaly
            max_anomaly_len (int): Maximum length of an anomaly
            window_slide_step (int): Size of each window slide step when determining intervals
            window_size_step (int): Size of each window increase step when determining intervals
            assume_independent (bool): If True, assumes independence between points for speed. False is not yet implemented
            which_metric (str): Metric to use for evaluating anomaly. Options are 'nlpd', 'msll', 'rmse', 'mll', or default is 'll' (log-likelihood)
            initial_lengthscale (float): Initial lengthscale for GP model. If None, no lengthscale is used (default) and the theta parameter is the identity matrix
        """

        # Initialize
        self.x = x
        self.y = y
        self.y_err = y_err
        self.min_anomaly_len = min_anomaly_len
        self.max_anomaly_len = max_anomaly_len
        self.window_slide_step = window_slide_step
        self.window_size_step = window_size_step
        self.assume_independent = assume_independent
        self.which_metric = which_metric
        self.initial_lengthscale = initial_lengthscale
        self.num_steps = len(x)
        self.anomalous = np.zeros(self.num_steps) # 0 means non-anomalous, 1 means anomalous at that time step

        # Check that min_anomaly_len is at least 1 and that max_anomaly_len is at least min_anomaly_len
        if min_anomaly_len < 1:
            raise ValueError("min_anomaly_len must be at least 1")
        if max_anomaly_len < min_anomaly_len:
            raise ValueError("max_anomaly_len must be at least min_anomaly_len")

        # Create original copies of x, y, and y_err
        self.x_orig = np.copy(x)
        self.y_orig = np.copy(y)
        self.y_err_orig = np.copy(y_err)

        # Store anomaly indicators (0 = non-anomalous, 1 = anomalous)
        self.anomalous = np.zeros(self.num_steps)

        # Possible anomaly intervals
        self.intervals = []
        for start in range(0, self.num_steps - min_anomaly_len, window_slide_step):
            for end in range(start + min_anomaly_len, min(start + max_anomaly_len, self.num_steps), window_size_step):
                self.intervals.append((start, end))
        
        self.mean_metrics = []

    def find_anomalous_interval(self, device=torch.device("cpu"), training_iterations=10, filename="", silent=True):
        # Initialize
        self.min_metric = np.inf
        self.best_interval = None

        if filename == "":
            save_to_txt = False
        else:
            # Create txt file to save results
            save_to_txt = True

            # write header
            with open(filename, 'w') as f:
                f.write('start,end,metric\n')

        # Iterate over each possible anomaly interval
        for start, end in self.intervals: 
            metric_sum = 0

            # Create train data without interval
            mask = np.ones(self.num_steps, dtype=bool)
            mask[start:end] = False
            x_train = torch.tensor(self.x[mask], dtype=torch.float32).to(device)
            y_train = torch.tensor(self.y[mask], dtype=torch.float32).to(device)
            y_err_train = torch.tensor(self.y_err[mask], dtype=torch.float32).to(device) if self.y_err is not None else None

            # Create test data with interval
            mask = ~mask
            x_test = torch.tensor(self.x[mask], dtype=torch.float32).to(device)
            y_test = torch.tensor(self.y[mask], dtype=torch.float32).to(device)
            y_err_test = torch.tensor(self.y_err[mask], dtype=torch.float32).to(device) if self.y_err is not None else None

            # Train GP model on train data
            model, likelihood, mll = train_gp(x_train, y_train, y_err_train, lengthscale=self.initial_lengthscale, training_iterations=training_iterations, device=device)

            # Evaluate metric for prediction on test data
            model.eval()
            likelihood.eval()

            # For each point in the interval, calculate the metric and sum them up
            for i in range(end - start):
                x_curr = x_test[i].unsqueeze(0)
                y_curr = y_test[i].unsqueeze(0)

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    f_pred = model(x_curr)
                    y_pred = likelihood(f_pred)

                if self.which_metric == 'nlpd':
                    metric = gpytorch.metrics.negative_log_predictive_density(y_pred, y_curr)

                elif self.which_metric == 'msll':
                    metric = gpytorch.metrics.mean_standardized_log_loss(y_pred, y_curr)

                elif self.which_metric == 'rmse':
                    pred_mean = y_pred.mean.cpu().numpy()
                    metric = np.sqrt(np.mean((pred_mean - y_curr.cpu().numpy())**2))

                elif self.which_metric == 'mll':
                    metric = mll(f_pred, y_curr)

                else: # Default to log-likelihood
                    metric = y_pred.log_prob(y_curr)

                metric_sum += metric

            metric_mean = metric_sum / (end - start)
            self.mean_metrics.append(metric_mean)

            # Update best interval
            if metric_mean < self.min_metric:
                self.min_metric = metric_mean
                self.best_interval = (start, end)

            if not silent:
                print(f"Anomaly interval: {start}-{end}, mean metric over the interval: {metric_mean}")

            # Save results to txt if save_to_txt is True
            if save_to_txt:
                with open(filename, 'a') as f:
                    f.write(f'{start},{end},{metric_mean}\n')

            # Delete all variables to free up memory
            del model
            del likelihood
            del mll
            del y_pred
            del f_pred
            del metric_mean
            del metric_sum
            del mask
            del x_train
            del y_train
            del y_err_train
            torch.cuda.empty_cache()
            gc.collect()

    