import gpytorch
import torch
import matplotlib.pyplot as plt

from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import Kernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# Create a Quasi-Periodic Kernel
class QuasiPeriodicKernel(Kernel):
    def __init__(self, periodic_kernel=None, rbf_kernel=None, **kwargs):
        super(QuasiPeriodicKernel, self).__init__(**kwargs)
        if periodic_kernel is None:
            self.periodic_kernel = PeriodicKernel(**kwargs)
        else:
            self.periodic_kernel = periodic_kernel
        
        if rbf_kernel is None:
            self.rbf_kernel = RBFKernel(**kwargs)
        else:
            self.rbf_kernel = rbf_kernel

    def forward(self, x1, x2, diag=False, **params):
        periodic_part = self.periodic_kernel.forward(x1, x2, diag=diag, **params)
        rbf_part = self.rbf_kernel.forward(x1, x2, diag=diag, **params)
        return periodic_part * rbf_part

# Create a GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, mean):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Define GP model with parameterized kernel
class ParameterizedGPModel(gpytorch.models.GP):
    def __init__(self, kernel, mean, likelihood):
        super().__init__()
        self.mean_module = mean
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Function to train GP model
def train_gp(
        x_train, 
        y_train, 
        training_iterations=1_000, 
        lr=0.01,
        device=torch.device("cpu"), 
        which_metric="mll",
        x_valid=None,
        y_valid=None,
        likelihood=None,
        kernel=None,
        mean=None,
        early_stopping=True,
        min_iterations=None,
        patience=1,
        plot=False
    ):

    if which_metric not in ["mll", "mse"]:
        raise ValueError("which_metric must be either 'mll' or 'mse'.")

    if early_stopping and min_iterations is None:
        min_iterations = training_iterations // 10
        print(f"Using {min_iterations} as minimum iterations for early stopping.")

    if y_valid is None:
        print("No validation data provided. Using training data for validation.")
        x_valid = x_train
        y_valid = y_train

    # Initialize likelihood
    if likelihood is not None:
        likelihood = likelihood.to(device)
        
    else:
        print("No likelihood specified. Using Gaussian likelihood.")
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # Initialize model
    if kernel is not None:
        model = ExactGPModel(
            x_train, 
            y_train, 
            likelihood, 
            kernel,
            mean
        ).to(device)

    else:
        print("No kernel specified. Using Quasi-Periodic Kernel with no priors or constraints.")
        model = ExactGPModel(
            x_train, 
            y_train, 
            likelihood,
            ScaleKernel(QuasiPeriodicKernel()),
            ConstantMean()            
        ).to(device)

    # Find optimal hyperparameters
    model.train()
    likelihood.train()

    # Set optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Plot loss during training
    train_losses = []
    valid_losses = []
    increase_count = 0

    for i in range(training_iterations):
        optimizer.zero_grad()
        pred = model(x_train)

        model.eval()
        with torch.no_grad():
            valid_pred = model(x_valid)
        model.train()

        # Compute losses
        if which_metric == "mse":
            train_loss = torch.nn.functional.mse_loss(pred.mean, y_train)
            valid_loss = torch.nn.functional.mse_loss(valid_pred.mean, y_valid)
        else:
            train_loss = -mll(pred, y_train)
            valid_loss = -mll(valid_pred, y_valid)

        # Early stopping
        if early_stopping:
            if i > min_iterations and valid_loss - valid_losses[i-1] > 0:
                increase_count += 1
            
            if increase_count > patience:
                print(f"Early stopping at iteration {i} due to increasing valid loss.")
                break

        train_loss.backward()
        optimizer.step()

        # Save losses
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())

    if plot:
        # Plot the train and valid loss side by side 
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(train_losses)), train_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.title("Train Loss for metric " + which_metric)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(valid_losses)), valid_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Valid Loss")
        plt.title("Valid Loss for metric " + which_metric)
        plt.show()

        # Plot the covariance matrices
        with torch.no_grad():
            periodic_cov = model.covar_module.base_kernel.periodic_kernel(torch.tensor(x_train).to(device)).evaluate()
            rbf_cov = model.covar_module.base_kernel.rbf_kernel(torch.tensor(x_train).to(device)).evaluate()
            cov_matrix = model.covar_module(torch.tensor(x_train).to(device)).evaluate()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Covariance Matrix")
        plt.imshow(cov_matrix.cpu().numpy(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Periodic Kernel Covariance")
        plt.imshow(periodic_cov.cpu().numpy(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("RBF Kernel Covariance")
        plt.imshow(rbf_cov.cpu().numpy(), cmap='viridis')
        plt.colorbar()

    return model, likelihood, mll