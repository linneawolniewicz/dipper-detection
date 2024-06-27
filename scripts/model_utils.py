import gpytorch
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# Define GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Function to train GP model
def train_gp(x_train, y_train, y_err_train, training_iterations=50, lengthscale=None, device=torch.device("cpu")):
    # Removed using the errors to inform the Gaussian Noise because it leads to numerical instabilities
    # noise_variances = y_err_train ** 2
    # min_noise_variance = 0.0001
    # noise = torch.clamp(noise_variances, min=min_noise_variance)
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise).to(device)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood).to(device)

    # Set lengthscale if given
    if lengthscale is not None:
        model.covar_module.base_kernel.lengthscale = torch.ones_like(model.covar_module.base_kernel.lengthscale) * lengthscale
    
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