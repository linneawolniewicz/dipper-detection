import gpytorch
import torch
from gpytorch.kernels import Kernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# Create a Quasi-Periodic Kernel
class QuasiPeriodicKernel(Kernel):
    def __init__(self, **kwargs):
        super(QuasiPeriodicKernel, self).__init__(**kwargs)
        self.periodic_kernel = PeriodicKernel()
        self.rbf_kernel = RBFKernel()

    def forward(self, x1, x2, diag=False, **params):
        periodic_part = self.periodic_kernel.forward(x1, x2, diag=diag, **params)
        rbf_part = self.rbf_kernel.forward(x1, x2, diag=diag, **params)
        return periodic_part * rbf_part

# Define GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(QuasiPeriodicKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Function to train GP model
def train_gp(x_train, y_train, y_err_train, training_iterations=50, lengthscale=None, device=torch.device("cpu")):
    # If this gives numerical warnings, use GaussianLikelihood() instead of FixedNoiseGaussianLikelihood()
    noise_variances = y_err_train ** 2
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise_variances, 
        learn_additional_noise=True
    ).to(device)

    # Initialize likelihood and model
    model = ExactGPModel(x_train, y_train, likelihood).to(device)

    # Set lengthscale if given
    if lengthscale is not None:
        model.covar_module.base_kernel.rbf_kernel.lengthscale = torch.ones_like(model.covar_module.base_kernel.rbf_kernel.lengthscale) * lengthscale
    
    # Find optimal hyperparameters
    model.train()
    likelihood.train()

    # Set optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        return model, likelihood, mll