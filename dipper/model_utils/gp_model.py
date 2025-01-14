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
        likelihood=None,
        kernel=None,
        mean=None,
        early_stopping=True,
        plot=False
    ):

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
    mll_losses = []
    mse_losses = []
    increase_count = 0
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(x_train)
        mll_loss = -mll(output, y_train)
        mse_loss = torch.nn.functional.mse_loss(output.mean, y_train)

        # Early stopping
        if early_stopping:
            if i > 300 and mll_loss - mll_losses[i-1] > 0:
                increase_count += 1
            
            if increase_count > 5:
                print(f"Early stopping at iteration {i} due to increasing loss.")
                break

        mll_loss.backward()
        optimizer.step()
        mll_losses.append(mll_loss.item())
        mse_losses.append(mse_loss.item())

    if plot:
        # Plot the loss
        plt.figure(figsize=(10,5))
        plt.plot(range(len(mll_losses)), mll_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("MLL Loss")
        plt.show()

        # Plot the loss
        plt.figure(figsize=(10,5))
        plt.plot(range(len(mse_losses)), mse_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("MSE Loss")
        plt.show()

        # Plot the covariance matrices
        with torch.no_grad():
            periodic_cov = model.covar_module.base_kernel.periodic_kernel(torch.tensor(x).to(device)).evaluate()
            rbf_cov = model.covar_module.base_kernel.rbf_kernel(torch.tensor(x).to(device)).evaluate()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Periodic Kernel Covariance")
        plt.imshow(periodic_cov.cpu().numpy(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("RBF Kernel Covariance")
        plt.imshow(rbf_cov.cpu().numpy(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("RBF * Periodic Kernel Covariance")
        plt.imshow(rbf_cov.cpu().numpy() * periodic_cov.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.show()

    return model, likelihood, mll