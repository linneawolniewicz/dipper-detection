import gpytorch
import torch
import matplotlib.pyplot as plt

from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import Kernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import Interval, GreaterThan

def dom_period_initialization(dominant_period):
    # Define kernel initial values based on the dominant period scale
    if dominant_period < 1:
        period_length_constraint = Interval(lower_bound=0.4, upper_bound=5, initial_value=dominant_period)
        periodic_lengthscale_constraint = GreaterThan(lower_bound=0.1, initial_value=dominant_period)
        rbf_lengthscale_constraint = GreaterThan(lower_bound=1, initial_value=dominant_period * 3)

    elif dominant_period >= 1 and dominant_period < 4:
        period_length_constraint = Interval(lower_bound=0.4, upper_bound=5, initial_value=dominant_period)
        periodic_lengthscale_constraint = GreaterThan(lower_bound=0.2, initial_value=dominant_period / 2)
        rbf_lengthscale_constraint = GreaterThan(lower_bound=1, initial_value=dominant_period * 2)

    elif dominant_period >= 4 and dominant_period < 8:
        period_length_constraint = Interval(lower_bound=dominant_period - 1, upper_bound= dominant_period + 1, initial_value=dominant_period)
        periodic_lengthscale_constraint = GreaterThan(lower_bound=0.4, initial_value=dominant_period / 2)
        rbf_lengthscale_constraint = GreaterThan(lower_bound=dominant_period/3, initial_value=dominant_period * 1.5)

    else:
        period_length_constraint = Interval(lower_bound=dominant_period - 2, upper_bound= dominant_period + 2, initial_value=dominant_period)
        periodic_lengthscale_constraint = GreaterThan(lower_bound=0.4, initial_value=dominant_period)
        rbf_lengthscale_constraint = GreaterThan(lower_bound=dominant_period/4, initial_value=dominant_period)

    # Define the GP model
    kernel = QuasiPeriodicKernel(
        periodic_kernel=PeriodicKernel(
            period_length_constraint=period_length_constraint, 
            lengthscale_constraint=periodic_lengthscale_constraint
        ),
        rbf_kernel=RBFKernel(
            lengthscale_constraint=rbf_lengthscale_constraint
        )
    )

    return kernel

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
    def __init__(self, kernel, mean):
        super().__init__()
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def train_gp(
        x_train, 
        y_train, 
        training_iterations=1_000, 
        lr=0.01,
        device=torch.device("cpu"), 
        which_metric="mll",
        likelihood=None,
        kernel=None,
        mean=None,
        which_opt='adam',
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

    # Initialize likelihood
    if likelihood is not None:
        likelihood = likelihood.to(device)
        
    else:
        print("No likelihood specified. Using Gaussian likelihood.")
        likelihood = GaussianLikelihood().to(device)

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
    if which_opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif which_opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("which_opt must be either 'adam' or 'sgd'.")
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Plot loss during training
    train_losses = []
    increase_count = 0

    for i in range(training_iterations):
        optimizer.zero_grad()
        pred = model(x_train)

        # Compute losses
        if which_metric == "mse":
            train_loss = torch.nn.functional.mse_loss(pred.mean, y_train)
        else:
            train_loss = -mll(pred, y_train)

        # Early stopping
        if early_stopping:
            if i > min_iterations and train_loss - train_losses[i-1] > 0:
                increase_count += 1
            
            if increase_count > patience:
                print(f"Early stopping at iteration {i} due to increasing train loss.")
                break

        train_loss.backward()
        optimizer.step()

        # Save losses
        train_losses.append(train_loss.item())

    if plot:
        # Plot the train loss 
        plt.figure(figsize=(5,5))
        plt.plot(range(len(train_losses)), train_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.title("Train Loss for metric " + which_metric)

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