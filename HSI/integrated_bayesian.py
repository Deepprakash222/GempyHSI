"""
MyModel: Pyro model for Integerated Bayesian method.

Used both for coordinate-only synthetic data (obs_data = the label column
directly) and for hyperspectral data with a GMM-derived pseudo-label
(obs_data = that pseudo-label). The likelihood is a simple Gaussian
comparing GemPy's continuous output to obs_data.
"""
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule

from prior import sample_interface_priors_and_update

import gempy as gp
from gempy_engine.core.backend_tensor import BackendTensor
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

class MyModel_IB(PyroModule):
    def __init__(self):
        super(MyModel_IB, self).__init__()

    def posterior(self, 
                  obs_data,
                  random_variable,
                  geo_model_test,
                  num_layers,
                  config,
                  dtype,
                  device):
            
        """
        Args:
            obs_data: Observed labels (synthetic ground truth, or a
                GMM-derived pseudo-label for real hyperspectral data).
            random_variable: List of interface-update specs -- which
                surface points are random variables and their priors.
            geo_model_test: GemPy model to update and recompute.
            num_layers: Number of interface points being inferred.
            config: Configuration object containing model parameters.
            dtype, device: Torch dtype/device.
        """

        
        custom_grid_values = sample_interface_priors_and_update(
                    random_variable,
                    geo_model_test,
                    num_layers,
                    dtype,
                    device,
                    enforce_ordering=config.enforce_ordering
                )
        
        mean = []
        cov = []
        
        z_nk = F.softmax(- config.scale* (torch.linspace(1,config.clusters,config.clusters, device =device, dtype =dtype) - custom_grid_values.reshape(-1,1))**2, dim=1)
        N_k = torch.sum(z_nk,axis=0)
        N = len(custom_grid_values)
        pi_k = N_k /N
        
        ################# Condition 1:  Deterministic for mean and covariance for hsi data ###########################
        for i in range(z_nk.shape[1]):
            mean_k = torch.sum( z_nk[:,i][:,None] * obs_data, axis=0)/ N_k[i]
            cov_k = torch.zeros((mean_k.shape[0],mean_k.shape[0]),device =device, dtype =dtype)
            for j in range(z_nk.shape[0]):
                cov_k +=  z_nk[j,i]* torch.matmul((obs_data[j,:] - mean_k).reshape((-1,1)) ,(obs_data[j,:] - mean_k).reshape((1,-1)))
            mean.append(mean_k)
            cov_k=cov_k/N_k[i] #+ 1e-3 * torch.diag(torch.ones(cov_k.shape[0],device =device, dtype =dtype))
            cov.append(cov_k)
            
        mean_tensor = torch.stack(mean, dim=0)
        cov_tensor = torch.stack(cov, dim=0)
        
        # --- Diagnostic: check for NaN and print everything relevant ---
        if torch.isnan(cov_tensor).any() or torch.isnan(mean_tensor).any():
            print("=" * 60)
            print("custom grid:", custom_grid_values)
            print("z_nk:", z_nk)
            print("NaN DETECTED in mean_tensor and/or cov_tensor")
            print("N_k (per-cluster total weight):", N_k)
            print("Clusters with N_k == 0:", torch.where(N_k == 0)[0].tolist())
            print("Clusters with N_k < 1e-6 (near-zero):", torch.where(N_k < 1e-6)[0].tolist())
            print("z_nk (first 10 rows):\n", z_nk[:10])
            print("z_nk column sums (should equal N_k):", z_nk.sum(dim=0))
            print("mean_tensor:\n", mean_tensor)
            print("cov_tensor:\n", cov_tensor)
            print("custom_grid_values range:", custom_grid_values.min().item(), "to", custom_grid_values.max().item())
            print("=" * 60)
            exit()

        gaussian_density_individual = torch.exp(dist.MultivariateNormal(mean_tensor, cov_tensor).log_prob(obs_data.unsqueeze(1)))  # (N, K)
        
        log_likelihood= torch.log(torch.sum(pi_k.unsqueeze(0) * gaussian_density_individual, axis=1))
        
        # Always write pyro.factor outside the pyro plate, otherwise the likelihood will be multiplied with number of plate dim
        pyro.factor("log_likelihood", log_likelihood.sum())  # Scalar log joint 
        