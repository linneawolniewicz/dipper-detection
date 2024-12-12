import gpytorch
import torch
import numpy as np
from scipy.stats import lognorm
from sklearn.mixture import GaussianMixture
from gpytorch.means import ConstantMean
from gpytorch.kernels import Kernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# Create parameterized kernel
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

# Create parameterized model
class ParameterizedGPModel(gpytorch.models.GP):
    def __init__(self, mean_constant, outputscale, noise_vars, period_length, periodic_lengthscale, rbf_lengthscale):
        super().__init__()
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant = mean_constant

        kernel = ParameterizedQuasiPeriodicKernel(period_length, periodic_lengthscale, rbf_lengthscale)
        
        self.covar_module = ScaleKernel(kernel)
        self.covar_module.outputscale = outputscale

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise_vars)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def generate_synthetic_lc_sample(
    num_timesteps=3000,
    device="cpu"
    ):
    # Sample GP parameters
    params = sample_parameters()
    period_length = torch.tensor(params["period_length"]).to(device)
    periodic_lengthscale = torch.tensor(params["periodic_lengthscale"]).to(device)
    rbf_lengthscale = torch.tensor(params["rbf_lengthscale"]).to(device)
    outputscale = torch.tensor(params["outputscale"]).to(device)
    mean_constant = torch.tensor(params["mean_constant"]).to(device)

    # Get sample
    x = torch.linspace(0, 10, num_timesteps).to(device)

    # Create noises of same length as x
    noise_vars = torch.tensor([params["noise"]] * num_timesteps).to(device)

    model = ParameterizedGPModel(mean_constant, outputscale, noise_vars, period_length, periodic_lengthscale, rbf_lengthscale).to(device)

    # Generate predictions
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        # Generate the GP predictions
        predictions = model.likelihood(model(x))
        sampled_gp = predictions.sample()

    # Convert to numpy for further processing
    x = x.detach().cpu().numpy()
    sampled_gp = sampled_gp.detach().cpu().numpy()

    # Add some high residuals randomly distributed
    num_high_residuals = len(params["high_residuals"])
    high_residual_indices = np.random.choice(num_timesteps, num_high_residuals, replace=False)
    for i, idx in enumerate(high_residual_indices):
        sampled_gp[idx] += params["high_residuals"][i]

    return x, sampled_gp
    
