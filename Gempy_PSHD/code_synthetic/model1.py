
import torch
import pyro
import pyro.distributions as dist

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor


from pyro.nn import PyroModule

# Change the backend to PyTorch for probabilistic modeling
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

class MyModel(PyroModule):
    def __init__(self):
        super(MyModel, self).__init__()
    
    #@config_enumerate
    def model_test(self, obs_data,interpolation_input_,geo_model_test,num_layers,likelihood_std,  dtype, device):
            """
            This Pyro model represents the probabilistic aspects of the geological model.
            It defines a prior distribution for the top layer's location and
            computes the thickness of the geological layer as an observed variable.

            obs_data: represents the observed data features reduced from 204 to 10 using PCA
            interpolation_input_: represents the dictionary of random variables for surface parameters
            geo_model_test : gempy model
            mean_init : initial means from gmm
            cov_init : initial cov from gmm
            num_layers: represents the number of layers we want to include in the model
            posterior_condition (int, optional): posterior condition. Defaults to 2.
            scale (float):  scaling factor to generate probability for each voxel
            cluster (int): number of cluster in our dataset
            alpha (float): Parameter to control the covariance matrix of drawing a sample for mean
            beta (float): Parameter to control the covariance matrix of drawing a sample for covariance
            """


            Random_variable ={}

            interpolation_input = geo_model_test.interpolation_input
            
            # Create a random variable based on the provided dictionary used to modify input data of gempy
            counter=1
            for interpolation_input_data in interpolation_input_[:num_layers]:
                
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if interpolation_input_data["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if interpolation_input_data["prior_distribution"]=="normal":
                        mean = interpolation_input_data["normal"]["mean"]
                        std  = interpolation_input_data["normal"]["std"]
                        Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std))
                        
                    elif interpolation_input_data["prior_distribution"]=="uniform":
                        min = interpolation_input_data["uniform"]["min"]
                        max = interpolation_input_data["uniform"]["min"]
                        Random_variable["mu_"+ str(interpolation_input_data['id'])] = pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max))
                        #print(counter)
                        #counter=counter+1
                        
                    else:
                        print("We have to include the distribution")
                
                
                    # Check which co-ordinates direction we wants to allow and modify the surface point data
                    if interpolation_input_data["direction"]=="X":
                        interpolation_input.surface_points.sp_coords = torch.index_put(
                            interpolation_input.surface_points.sp_coords,
                            (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                            Random_variable["mu_"+ str(counter)])
                    elif interpolation_input_data["direction"]=="Y":
                        interpolation_input.surface_points.sp_coords = torch.index_put(
                            interpolation_input.surface_points.sp_coords,
                            (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                            Random_variable["mu_"+ str(counter)])
                    elif interpolation_input_data["direction"]=="Z":
                        interpolation_input.surface_points.sp_coords = torch.index_put(
                            interpolation_input.surface_points.sp_coords,
                            (interpolation_input_data["id"], torch.tensor([2])),
                            Random_variable["mu_"+ str(counter)])
                        
                    else:
                        print("Wrong direction")
                
                counter=counter+1
            
            
            for i in range(len(interpolation_input_)+1):
                if i==0:
                    pyro.sample(f'mu_{i+1} < mu_{i+1} + 2 * std', dist.Delta(torch.tensor(1.0, dtype=dtype, device =device)), obs=(Random_variable[f'mu_{i+1}'] < interpolation_input_[0]["normal"]["mean"] + 2 * interpolation_input_[0]["normal"]["std"]))
                elif i==len(interpolation_input_):
                    pyro.sample(f'mu_{i} > mu_{i} - 2 * std', dist.Delta(torch.tensor(1.0, dtype=dtype, device =device)), obs=(Random_variable[f"mu_{i}"] > interpolation_input_[-1]["normal"]["mean"] - 2 * interpolation_input_[-1]["normal"]["std"]))
                else:
                    pyro.sample(f'mu_{i} > mu_{i+1} ', dist.Delta(torch.tensor(1.0, dtype=dtype, device =device)), obs=(Random_variable[f"mu_{i}"] > Random_variable[f"mu_{i+1}"]))
                
            
            # Update the model with the new top layer's location
            
            
            # # Compute the geological model
            geo_model_test.solutions = gempy_engine.compute_model(
                interpolation_input=interpolation_input,
                options=geo_model_test.interpolation_options,
                data_descriptor=geo_model_test.input_data_descriptor,
                geophysics_input=geo_model_test.geophysics_input,
            )
            
            # Compute and observe the thickness of the geological layer
            custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
            
            log_likelihood = dist.Normal(custom_grid_values, likelihood_std).log_prob(obs_data)  # (N, K)
            
            # Always write pyro.factor outside the pyro plate, otherwise the likelihood will be multiplied with number of plate dim
            pyro.factor("log_likelihood", log_likelihood.sum())  # Scalar log joint 
            
                
