import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.inspect import get_dependencies
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

import scipy.io
from scipy.stats import zscore
from sklearn.manifold import TSNE

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import json

def calculate_average_entropy(responsibilities):
    """
    Calculate the average Shannon entropy of responsibilities for a GMM.
    
    Args:
        responsibilities (numpy array): An (N, K) array where each element gamma_nk is the responsibility of the
                                         k-th component for the n-th data point.
    
    Returns:
        float: The average Shannon entropy across all data points.
    """
    # Clip responsibilities to avoid log(0)
    responsibilities = np.clip(responsibilities, 1e-10, 1.0)
    
    # Calculate the entropy for each data point
    entropy_per_point = -np.sum(responsibilities * np.log(responsibilities), axis=1)
    
    # Return the average entropy
    return np.mean(entropy_per_point)

def calculate_entropy(mixing_coefficient):
    """
    Calculate the average Shannon entropy of responsibilities for a GMM.
    
    Args:
        responsibilities (numpy array): An (N, K) array where each element gamma_nk is the responsibility of the
                                         k-th component for the n-th data point.
    
    Returns:
        float: The average Shannon entropy across all data points.
    """
    # Clip responsibilities to avoid log(0)
    mixing_coefficient = np.clip(mixing_coefficient, 1e-10, 1.0)
    
    # Calculate the entropy for each data point
    entropy_per_point = -np.sum(mixing_coefficient * np.log(mixing_coefficient))
    
    # Return the average entropy
    return entropy_per_point


def compute_map(posterior_samples,geo_model_test,label_data,test_list,num_layers, directory_path,likelihood_std, dtype, device):
    """
    This function computes the maximum a priori based on the posterior samples

    Args:
        posterior_samples : samples generated from mcmc
        geo_model_test : gempy model
        normalised_hsi : normalised hsi data of 204 dimensions as a tensor
        test_list : dictionary of surface points 
        y_obs_label : label data 
        mean_init : initial means from gmm
        cov_init : initial cov from gmm 
        directory_path : path to save file
        num_layers (int, optional): number of layers Defaults to 4.
        posterior_condition (int, optional): posterior condition. Defaults to 2.
        scale (float):  scaling factor to generate probability for each voxel
        cluster (int): number of cluster in our dataset
        alpha (float): Parameter to control the covariance matrix of drawing a sample for mean
        beta (float): Parameter to control the covariance matrix of drawing a sample for covariance
    """
    

    directory_path_MAP = directory_path +"/MAP"
    
    # Check if the directory exists
    if not os.path.exists(directory_path_MAP):
        # Create the directory if it does not exist
        os.makedirs(directory_path_MAP)
        print(f"Directory '{directory_path_MAP}' was created.")
    else:
        print(f"Directory '{directory_path_MAP}' already exists.")
    
    
    unnormalise_posterior_value={}
    store_accuracy=[]
    store_gmm_accuracy = []
    store_z_nk_entropy =[]
    store_gmm_entropy=[]
    store_mixing_entropy=[]
    
    # Convert the tensors to lists
    posterior_samples_serializable = {k: v.tolist() for k, v in posterior_samples.items()}
    
    
        
        
    unnormalise_posterior_value["log_prior_geo_list"]=[]
    unnormalise_posterior_value["log_likelihood_list"]=[]
    unnormalise_posterior_value["log_posterior_list"] =[]

    keys_list = list(posterior_samples.keys())

    ########## TODO###############################################################
    # Extend this to other distribution too
    ###############################################################################
    prior_mean_surface = [item['normal']['mean'].item() for item in test_list[:num_layers]]
    prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
    ###############################################################################

    RV_post_mu ={}

    # Get index of the samples in posterior
    for i in range(posterior_samples["mu_1"].shape[0]):
        # Get the georemtrical random variable for a given sample 
        
        for j in range(num_layers):  
            RV_post_mu[f"mu_{j+1}"] = posterior_samples[keys_list[j]][i]
            
        
        # Calculate the log probability of the value
        log_prior_geo = torch.tensor(0.0, dtype=dtype, device =device)
        for l in range(num_layers):
            log_prior_geo += dist.Normal(prior_mean_surface[l], prior_std_surface[l]).log_prob(RV_post_mu[f"mu_{l+1}"])
        ##########################################################################
        # Update the model with the new top layer's location
        ##########################################################################
        interpolation_input = geo_model_test.interpolation_input
        
        counter1=1
        for interpolation_input_data in test_list[:num_layers]:
            interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
            counter1=counter1+1
        
        
        
        # # Compute the geological model
        geo_model_test.solutions = gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=geo_model_test.interpolation_options,
            data_descriptor=geo_model_test.input_data_descriptor,
            geophysics_input=geo_model_test.geophysics_input,
        )
        
        # Compute and observe the thickness of the geological layer
        
        custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        
        log_likelihood = dist.Normal(custom_grid_values, likelihood_std).log_prob(label_data)  # (N, K)

        unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
        unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood.sum())
        unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo  + log_likelihood.sum())
            
    MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
    
    filename_posterior_samples =directory_path + "/posterior_samples.json"
    # Save to a JSON file
    with open(filename_posterior_samples, 'w') as f:
        json.dump(posterior_samples_serializable, f)
    return MAP_sample_index, unnormalise_posterior_value["log_posterior_list"][MAP_sample_index] , [ele.detach() for ele in unnormalise_posterior_value["log_posterior_list"]  ] 