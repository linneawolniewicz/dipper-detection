import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from gp_model import train_gp
from utils import check_identified_anomalies

# TODO: make this work when y_err is not provided (i.e. the synthetic lc)
class GPDetectAnomaly:
    def __init__(
            self, 
            x, 
            y, 
            y_err, 
            which_metric='mll', # 'nlpd', 'msll', 'rmse', or default is 'mll'
            num_anomalies=1, # Number of anomalies to detect
            initial_lengthscale=None, # If None, no lengthscale is used (default) and the theta parameter is the identity matrix
            expansion_param=1 # how many indices left and right to increase anomaly by
        ):

        self.x = x
        self.y = y
        self.y_err = y_err

        # Create copies of the original data
        self.x_orig = np.copy(x)
        self.y_orig = np.copy(y)
        self.y_err_orig = np.copy(y_err)

        # Initialize variables
        self.which_metric = which_metric 
        self.num_anomalies = num_anomalies 
        self.num_steps = len(x)
        self.anomalous = np.zeros(self.num_steps) # 0 means non-anomalous, 1 means anomalous at that time step
        self.initial_lengthscale = initial_lengthscale 
        self.expansion_param = expansion_param 

    def detect_anomaly(
            self, 
            training_iterations=30, 
            device='cpu', 
            plot=False, 
            anomaly_locs=None,
            detection_range=None,
        ):
        '''
        Method:
            1. Perform GP regression on the timeseries.
            2. Find the most significant outlier point.
            3. Exclude that point and redo regression. See if GP improves by some threshold.
            4. Exclude adjacent points and redo step 3.
            5. Repeat step 4 as long as GP improves the fit by some threshold.
            6. If no improvement, define anomaly signal as the difference between data and regression in that interval of points.
            7. Repeat steps 2-6 for a defined number of anomalies.
        '''
        # Train GP model
        model, likelihood, mll = train_gp(
                    torch.tensor(self.x, dtype=torch.float32).to(device),
                    torch.tensor(self.y, dtype=torch.float32).to(device), 
                    torch.tensor(self.y_err, dtype=torch.float32).to(device), 
                    training_iterations=training_iterations, 
                    lengthscale=self.initial_lengthscale, 
                    device=device
        )
        self.final_lengthscale = model.covar_module.base_kernel.rbf_kernel.lengthscale.item()

        # Step 7 (repeat for every anomaly)
        for i in range(self.num_anomalies):
            # Get subset of data that is flagged an non-anomalous
            x_sub = torch.tensor(self.x[self.anomalous == 0], dtype=torch.float32).to(device)
            y_sub = torch.tensor(self.y[self.anomalous == 0], dtype=torch.float32).to(device)
            y_err_sub = torch.tensor(self.y_err[self.anomalous == 0], dtype=torch.float32).to(device)

            # Re-fit the GP on non-anomalous data
            model, likelihood, mll = train_gp(x_sub, y_sub, y_err_sub, training_iterations=1, lengthscale=self.final_lengthscale, device=device)

            # Predict
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(torch.tensor(self.x, dtype=torch.float32).to(device)))
                pred_mean = observed_pred.mean.cpu().numpy()

            # Find index of largest deviation
            sig_dev = (pred_mean - self.y) / self.y_err
            non_anomalous_indices = np.where(self.anomalous == 0)[0]  # Indices where anomalous == 0
            index = non_anomalous_indices[np.argmax(sig_dev[self.anomalous == 0])]  # Correct index in original array
            assert self.anomalous[index] == 0, f"Anomaly index {index} is already flagged as anomalous"
            print(f"\n\n New dip identified at anomalous index {index}, x[index] = {self.x[index]}, anomalous[index] = {self.anomalous[index]}")

            # Plot if desired
            if plot:
                fig, axs = plt.subplots(2, 1, sharex = True, figsize = (8, 8))
                axs[0].set_title("GP Mean Prediction vs. Data")
                axs[0].plot(self.x, pred_mean, "grey", lw=2, label="Prediction on all Data")
                axs[0].plot(self.x[(self.anomalous==0)], self.y_orig[(self.anomalous==0)], '.k', markersize=2, label="Flagged non-anomalous Data")
                axs[0].plot(self.x[(self.anomalous==1)], self.y_orig[(self.anomalous==1)], '.r', markersize=2, label="Flagged anomalous Data")
                axs[0].set_ylim(np.min(y_sub.cpu().numpy()), np.max(y_sub.cpu().numpy()))
                axs[0].legend()

                axs[1].set_title("Significance of Deviation from Original Data")
                axs[1].plot(self.x[(self.anomalous==0)], np.abs(sig_dev[(self.anomalous==0)]), '.k', markersize=2, label="Flagged non-anomalous Data")
                axs[1].plot(self.x[(self.anomalous==1)], np.abs(sig_dev[(self.anomalous==1)]), '.r', markersize=2, label="Flagged anomalous Data")

                if anomaly_locs is not None:
                    # Plot the anomaly_range in axs[0]
                    for i in range(len(anomaly_locs)):
                        anomaly_range = np.arange(int(anomaly_locs[i]) - int(detection_range), int(anomaly_locs[i]) + int(detection_range))
                        axs[0].axvspan(self.x[anomaly_range[0]], self.x[anomaly_range[-1]], color='gold')

                    # Plot the anomaly_range in axs[1]
                    for i in range(len(anomaly_locs)):
                        anomaly_range = np.arange(int(anomaly_locs[i]) - int(detection_range), int(anomaly_locs[i]) + int(detection_range))
                        axs[1].axvspan(self.x[anomaly_range[0]], self.x[anomaly_range[-1]], color='gold')

                # Plot the index of the new anomaly
                axs[1].axvline(x=self.x[index], color='b', linestyle='--', alpha=0.5, label="New flagged anomaly")
                axs[1].legend()
                plt.show(block=False)
                plt.close()

            # Intialize variables for expanding anomalous region
            left_edge = index
            right_edge = index
            diff_metric = 1e6
            metric = 1e6
            
            # While the metric is decreasing, expand the anomalous edges
            while diff_metric > 0:
                # Subset x, y, and y_err
                subset = (((np.arange(self.num_steps) > right_edge) | (np.arange(self.num_steps) < left_edge)) & (self.anomalous == 0))
                x_sub = torch.tensor(self.x[subset], dtype=torch.float32).to(device)
                y_sub = torch.tensor(self.y[subset], dtype=torch.float32).to(device)
                y_err_sub = torch.tensor(self.y_err[subset], dtype=torch.float32).to(device)
                
                # Re-fit the GP on non-anomalous data
                model, likelihood, mll = train_gp(x_sub, y_sub, y_err_sub, training_iterations=1, lengthscale=self.final_lengthscale, device=device)

                # Predict
                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    # Subset
                    observed_pred_sub = likelihood(model(x_sub))
                    pred_mean_sub = observed_pred.mean.cpu().numpy()
                    
                    # Full
                    observed_pred_full = likelihood(model(torch.tensor(self.x, dtype=torch.float32).to(device)))
                    pred_mean_full = observed_pred_full.mean.cpu().numpy()

                # Calculate metric difference
                old_metric = metric

                if self.which_metric == 'nlpd':
                    metric = gpytorch.metrics.negative_log_predictive_density(observed_pred_sub, y_sub)

                elif self.which_metric == 'msll':
                    metric = gpytorch.metrics.mean_standardized_log_loss(observed_pred_sub, y_sub)

                elif self.which_metric == 'rmse':
                    metric = np.sqrt(np.mean((pred_mean_sub - y_sub.cpu().numpy())**2))

                else: # metric == mll
                    model.eval()
                    likelihood.eval()
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        output = model(x_sub)
                        metric = mll(output, y_sub)

                diff_metric = old_metric - metric # smaller is better
                # print(f"Old metric: {old_metric} - New metric: {metric} = Diff metric: {diff_metric}")

                # Expand anomalous region on both sides
                if left_edge >= 0 + self.expansion_param:
                    left_edge -= self.expansion_param

                if right_edge < self.num_steps - self.expansion_param:
                    right_edge += self.expansion_param

                # print(f"Anomaly index {i} x[i] {self.x[i]}, left_edge:right_edge {left_edge}:{right_edge}")

            # Update anomalous array and remove anomalies from y
            self.y[left_edge:right_edge] = pred_mean_full[left_edge:right_edge]
            self.anomalous[left_edge:right_edge] = 1
            print(f"Anomalous edges = {left_edge}:{right_edge}")
    
    def predict_anomaly(
            self, 
            device='cpu', 
            plot=False, 
            save_name=None, 
            anomaly_locs=None, 
            min_contiguous=1,
            detection_range=None
        ):
        # Subset x, y, and y_err
        subset = (self.anomalous == 0)
        x_sub = torch.tensor(self.x[subset], dtype=torch.float32).to(device)
        y_sub = torch.tensor(self.y[subset], dtype=torch.float32).to(device)
        y_err_sub = torch.tensor(self.y_err[subset], dtype=torch.float32).to(device)

        # Fit on final subset
        model, likelihood, mll = train_gp(x_sub, y_sub, y_err_sub, training_iterations=1, lengthscale=self.final_lengthscale, device=device)

        # Predict
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(torch.tensor(self.x, dtype=torch.float32).to(device)))
            pred_mean = observed_pred.mean.cpu().numpy()
            pred_var = observed_pred.variance.cpu().numpy()

        # Plot if desired
        if plot==True or save_name is not None:
            fig, axs = plt.subplots()
            axs.set_title("GP Mean Prediction vs Data")
            axs.plot(self.x, pred_mean, "grey", lw=2, label="Prediction on all Data")
            axs.plot(x_sub.cpu().numpy(), y_sub.cpu().numpy(), '.k', markersize=2, label="Flagged non-anomalous Data")
            axs.set_ylim(np.min(self.y_orig), np.max(self.y_orig))
            axs.plot(self.x_orig[(self.anomalous==1)], self.y_orig[(self.anomalous==1)], '.r', markersize=2, label="Flagged anomalous Data")
            
            if anomaly_locs is not None:
                # Plot the anomaly_range
                for i in range(len(anomaly_locs)):
                    anomaly_range = np.arange(int(anomaly_locs[i]) - int(detection_range), int(anomaly_locs[i]) + int(detection_range))
                    axs.axvspan(self.x[anomaly_range[0]], self.x[anomaly_range[-1]], color='gold')
                axs.legend(loc='upper right')

            if plot:
                plt.show(block=True)

            if save_name is not None:
                plt.savefig(save_name)

            plt.close()

        # Check identified anomalies if anomaly_locs are provided
        if anomaly_locs is not None:
            identified, identified_ratio = check_identified_anomalies(
                anomaly_locs, 
                self.anomalous, 
                detection_range,
                min_contiguous
            )

            print(f"Injected anomaly centers: {anomaly_locs}")
            print(f"Anomalies identified: {identified}")
            print(f"Ratio of anomalies identified: {identified_ratio}")