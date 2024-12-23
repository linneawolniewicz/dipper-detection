import gpytorch
import torch
import matplotlib.pyplot as plt
from gpytorch.kernels import Kernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# Create a Quasi-Periodic Kernel
class QuasiPeriodicKernel(Kernel):
    def __init__(self, **kwargs):
        super(QuasiPeriodicKernel, self).__init__(**kwargs)
        self.periodic_kernel = PeriodicKernel(**kwargs)
        self.rbf_kernel = RBFKernel(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        periodic_part = self.periodic_kernel.forward(x1, x2, diag=diag, **params)
        rbf_part = self.rbf_kernel.forward(x1, x2, diag=diag, **params)
        return periodic_part * rbf_part

# Define GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Create a parameterized Quasi-Periodic Kernel
class ParameterizedQuasiPeriodicKernel(Kernel):
    def __init__(self, period_length, periodic_lengthscale, rbf_lengthscale):
        super().__init__()
        self.periodic_kernel = PeriodicKernel()
        self.periodic_kernel.period_length = period_length
        self.periodic_kernel.lengthscale = periodic_lengthscale

        self.rbf_kernel = RBFKernel()
        self.rbf_kernel.lengthscale = rbf_lengthscale

    def forward(self, x1, x2, diag=False, **params):
        periodic_part = self.periodic_kernel.forward(x1, x2, diag=diag, **params)
        rbf_part = self.rbf_kernel.forward(x1, x2, diag=diag, **params)
        return periodic_part * rbf_part

# Define GP model with parameterized kernel
class ParameterizedGPModel(gpytorch.models.GP):
    def __init__(self, kernel, mean_constant, outputscale, likelihood):
        super().__init__()
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant = mean_constant
        
        self.covar_module = ScaleKernel(kernel)
        self.covar_module.outputscale = outputscale

        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Function to train GP model
def train_gp(
        x_train, 
        y_train, 
        y_err_train, 
        training_iterations=50, 
        lr=0.1,
        learn_additional_noise=True, 
        device=torch.device("cpu"), 
        kernel=None,
        plot=False
    ):
    # Create noisy likelihood
    noise_variances = y_err_train ** 2
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise_variances, 
        learn_additional_noise=learn_additional_noise
    ).to(device) # If this gives numerical warnings, use GaussianLikelihood() instead of FixedNoiseGaussianLikelihood()

    # Initialize model
    if kernel is not None:
        model = ExactGPModel(
            x_train, 
            y_train, 
            likelihood, 
            kernel,
        ).to(device)

    else:
        print("No kernel specified. Using Quasi-Periodic Kernel with no priors or constraints.")
        model = ExactGPModel(
            x_train, 
            y_train, 
            likelihood,
            QuasiPeriodicKernel()
        ).to(device)

    # Find optimal hyperparameters
    model.train()
    likelihood.train()

    # Set optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Plot loss during training
    losses = []
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if plot:
        # Plot the loss
        plt.figure(figsize=(10,5))
        plt.plot(range(training_iterations), losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
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