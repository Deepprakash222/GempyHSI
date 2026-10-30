"""
MyModel: Pyro model for the Pre-Segmented Hyperspectral Data (PSHD) method.

Used both for coordinate-only synthetic data (obs_data = the label column
directly) and for hyperspectral data with a GMM-derived pseudo-label
(obs_data = that pseudo-label). The likelihood is a simple Gaussian
comparing GemPy's continuous output to obs_data.
"""

import pyro
import pyro.distributions as dist
import gempy as gp
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from pyro.nn import PyroModule

from prior import sample_interface_priors_and_update

import gempy as gp
from gempy_engine.core.backend_tensor import BackendTensor
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)



class MyModel_PS(PyroModule):
    def __init__(self):
        super(MyModel_PS, self).__init__()

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
        
        #custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        
        
        log_likelihood = dist.Normal(custom_grid_values, config.likelihood_std).log_prob(obs_data)

        # pyro.factor must be outside any plate, or the likelihood gets
        # multiplied by the plate dimension.
        pyro.factor("log_likelihood", log_likelihood.sum())