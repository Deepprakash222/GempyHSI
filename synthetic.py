import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
from datetime import datetime
import json
import argparse

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.inspect import get_dependencies
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.mcmc.util import TraceEinsumEvaluator

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

import io
import contextlib
from code_synthetic.model1 import MyModel
from code_synthetic.map_func import compute_map
from code_synthetic.initial_gempy_model import *
from code_synthetic.final_gempy_model import *

parser = argparse.ArgumentParser(description='pass values using command line')
parser.add_argument('--cluster', metavar='cluster', type=int, default=3, help='total number of cluster')
parser.add_argument('--prior_number_samples', metavar='prior_number_samples', type=int , default=1000, help='number of samples for prior')
parser.add_argument('--posterior_number_samples', metavar='posterior_number_samples', type=int , default=250, help='number of samples for posterior')
parser.add_argument('--posterior_warmup_steps', metavar='posterior_warmup_steps', type=int , default=250, help='number of  warmup steps for posterior')
parser.add_argument('--directory_path', metavar='directory_path', type=str , default="./Results", help='name of the directory in which result should be stored')
parser.add_argument('--dataset', metavar='dataset', type=str , default="Syn_label_shift_20_error", help='name of the dataset (Syn_label, Syn_label_shift, Syn_label_shift_20_error or other)')
parser.add_argument('--posterior_num_chain', metavar='posterior_num_chain', type=int , default=5, help='number of chain')

parser.add_argument('--slope_gempy', metavar='slope_gempy', type=float , default=200.0, help='slope for gempy')

parser.add_argument('--scale', metavar='scale', type=float , default=10.0, help='scaling factor to generate probability for each voxel')
parser.add_argument('--likelihood_std', metavar='ikelihood_std', type=float , default=0.4, help='scaling parameter for the mean, 0.1')


dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cluster_acc(Y_pred, Y, ignore_label=None):
    """ Rearranging the class labels of prediction so that it maximise the 
        match class labels.

    Args:
        Y_pred (int): An array for predicted labels
        Y (float): An array for true labels
        ignore_label (int, optional): Laels to be ignored

    Returns:
       row (int): A list of index of row 
       column (int) : A list of index of column
       accuracy (float): accuracy after we found correct label
       cost_matrix (int) : cost matrix 
    """
    if ignore_label is not None:
        index = Y!= ignore_label
        Y=Y[index]
        Y_pred=Y_pred[index]
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.shape == Y.shape
    D = int((max(Y_pred.max(), Y.max())).item())
    w = torch.zeros((D, D))
    for i in range(Y_pred.shape[0]):
        w[int(Y_pred[i].item())-1, int(Y[i].item())-1] += 1
    ind = linear_assignment(w.max() - w)
    return ind[0], ind[1], (w[ind[0], ind[1]]).sum() / Y_pred.shape[0], w

def TSNE_transformation(data, label, filename):
    """ This function applies TSNE algorithms to reduce the high dimensional data into 2D
        for better visualization

    Args:
        data (float): High dimensional Input data 
        label (int): Label information of each data entry
        filename (str): Location to store the image after dimensionality reduction
    """
    from sklearn.manifold import TSNE
    model = TSNE(n_components=1, random_state=42)
    transformed_data = model.fit_transform(data[:,1:]) 
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    transformed_data = np.concatenate([data[:,0].reshape((-1,1)), transformed_data.reshape((-1,1))], axis=1)
    plt.figure(figsize=(10,8))
    for label_ in np.unique(label):
        idx =label ==label_
        plt.scatter(transformed_data[idx][:,0],transformed_data[idx][:,1], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('Component 2')
    plt.title("Data after dimensionality reduction")
    
    plt.savefig(filename)
    plt.close()

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

def objective(trial):
    # suggest hyperparameter
    alpha = trial.suggest_uniform("alpha", 1e-4, 100)
    beta  = trial.suggest_uniform("beta", 1e-4, 100)

def run_synthetic(dataset, geo_model_init, geo_model_final):
    """
    This function defines a model which uses hyperspectral data, applies clustering methods to find cluster information and then uses Bayesian
    """
    
    args = parser.parse_args()
    cluster = args.cluster
    prior_number_samples = args.prior_number_samples
    posterior_number_samples = args.posterior_number_samples
    posterior_warmup_steps = args.posterior_warmup_steps
    posterior_num_chain = args.posterior_num_chain
    directory_path = args.directory_path
    xyz_coord = dataset.data
    z = xyz_coord[:,2]
    slope_gempy = args.slope_gempy
    scale = args.scale
    likelihood_std = args.likelihood_std
    
    ## seed numpy and pytorch
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Setting the seed for Pyro sampling
    pyro.set_rng_seed(42)
    directory_path = directory_path + "/" + dataset.name  + "_slope_gempy_" + str(slope_gempy)  + "_likelihood_std_" + str(likelihood_std)
    
    # Check if the directory exists
   
    if not os.path.exists(directory_path):
        # Create the directory if it does not exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    ###########################################################################
    # 
    ###########################################################################
    labels = dataset.labels
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(labels):
        idx =labels ==label_
        plt.scatter(z[idx],labels[idx], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('label')
    plt.title("Dept vs label")
    filename = directory_path + "/ground_truth.png"
    plt.savefig(filename)
    plt.close()
    obs_data = torch.tensor(labels, dtype=torch.float64)
    

        
    # Create initial model with higher refinement for better resolution and save it
    prior_filename= directory_path + "/prior_model.png"
    geo_model_test = create_initial_gempy_model_KSL_3_layer(refinement=7,filename=prior_filename, save=True)
    # We can initialize again but with lower refinement because gempy solution are inddependent
    geo_model_test = create_initial_gempy_model_KSL_3_layer(refinement=3,filename=prior_filename, save=False)
    
    
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord.numpy())
        
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    geo_model_test.transform.apply_inverse(sp_coords_copy_test)
    geo_model_test.interpolation_options.sigmoid_slope = slope_gempy
    gp.compute_model(geo_model_test)
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()

    # Compute and observe the thickness of the geological layer 
    custom_grid_values_prior = torch.tensor(geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values, dtype=dtype, device=device)
    gempy_label = np.round(custom_grid_values_prior.numpy())
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(gempy_label):
        idx =gempy_label ==label_
        plt.scatter(z[idx],gempy_label[idx], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('label')
    plt.title("Dept vs label")
    filename = directory_path + "/Intial_model.png"
    plt.savefig(filename)
    plt.close()
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_init = geo_model_test.surface_points.df
    df_or_init = geo_model_test.orientations.df
    
    filename_initial_sp = directory_path + "/Initial_sp.csv"
    filename_initial_op = directory_path + "/Initial_op.csv"
    df_sp_init.to_csv(filename_initial_sp)
    df_or_init.to_csv(filename_initial_op)
    
    ################################################################################
   
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    geo_model_test.interpolation_options.sigmoid_slope = slope_gempy
    test_list=[]
    
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype, device=device), "std":torch.tensor(0.02,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype, device=device), "std":torch.tensor(0.02,dtype=dtype, device=device)}})
    
    num_layers = len(test_list)
    
    model = MyModel()
    
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    label_data =obs_data
    filename_Bayesian_graph =directory_path +"/Bayesian_graph.png"
    pyro.clear_param_store()
    dot = pyro.render_model(model.model_test, model_args=(label_data,test_list,geo_model_test,num_layers, likelihood_std, dtype, device),render_distributions=True,filename=filename_Bayesian_graph)
    #dot = pyro.render_model(model.model_test, model_args=(normalised_hsi,test_list,geo_model_test,mean_init,cov_init,factor,num_layers,posterior_condition, scale, cluster, alpha, beta,dtype,device))
    ################################################################################
    # Prior
    ################################################################################
    pyro.set_rng_seed(42)
    prior = Predictive(model.model_test, num_samples=prior_number_samples)(label_data,test_list,geo_model_test,num_layers, likelihood_std, dtype, device)
    # Key to avoid
    avoid_key =[]
    for i in range(len(test_list)+1):
                if i==0:
                    avoid_key.append(f'mu_{i+1} < mu_{i+1} + 2 * std')
                elif i==len(test_list):
                    avoid_key.append(f'mu_{i} > mu_{i} - 2 * std')
                else:
                    avoid_key.append(f'mu_{i} > mu_{i+1} ')
                    
    avoid_key.append('log_likelihood')
    #avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > mu_3', 'mu_3 > mu_4' , 'mu_4 > -83']
    # Create sub-dictionary without the avoid_key
    prior = dict((key, value) for key, value in prior.items() if key not in avoid_key)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    filename_prior_plot = directory_path + "/prior.png"
    plt.savefig(filename_prior_plot)
    plt.close()
    
    ################################################################################
    # Posterior 
    ################################################################################
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(model.model_test, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.75, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=posterior_number_samples,mp_context="spawn", warmup_steps=posterior_warmup_steps,num_chains=posterior_num_chain, disable_validation=False)
    mcmc.run(label_data,test_list,geo_model_test,num_layers, likelihood_std, dtype, device)
    
    #posterior_samples = mcmc.get_samples(group_by_chain=True)
    posterior_samples = mcmc.get_samples(group_by_chain=False)
    posterior_samples_ = mcmc.get_samples(group_by_chain=True)
    
    print("trace printing")
    # Trace the model with the current sample values
    import pyro.poutine as poutine
    
    # Calculate the log joint (which includes log likelihood and log prior)
    log_posterior_vals = []
    
    for sample in zip(*posterior_samples.values()):  # Iterate over samples of all parameters
        # Set the current values for the model parameters
        sample_dict = {name: value for name, value in zip(posterior_samples.keys(), sample)}
        
        # Trace the model with the current sample values and the data as an argument
        # You need to use `sample_dict` to assign values to the model's random variables
        trace = poutine.trace(lambda: model.model_test(label_data,test_list,geo_model_test,num_layers, likelihood_std, dtype, device)).get_trace()
        
        # Apply the sampled parameter values to the model trace
        for name, value in sample_dict.items():
            trace.nodes[name]["value"] = value  # Set the sampled values in the trace
        
        # we need to re run our model to update the value of node which are dependent on posterior sample
        replayed_model = poutine.replay(model.model_test, trace=trace)
        trace = poutine.trace(replayed_model).get_trace(label_data,test_list,geo_model_test,num_layers, likelihood_std, dtype, device)
        # Calculate the log joint (which includes log likelihood and log prior)
        log_joint = trace.log_prob_sum()
        
        log_posterior_vals.append(log_joint.item())  # .item() to extract scalar from tensor
    
    
    
    
    print("MCMC summary results")
    
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        mcmc.summary()  
        summary_output = buf.getvalue()

    
    print(summary_output)

    with open(f'{directory_path}/mcmc_summary_p.txt', 'w') as f:
        f.write(summary_output)
    
    posterior_samples_chain = {key: value[0] for key, value in posterior_samples_.items()}
    #print(posterior_samples_chain)
    posterior_predictive = Predictive(model.model_test, posterior_samples )(label_data,test_list,geo_model_test,num_layers, likelihood_std, dtype, device)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    filename_posteriro_plot = directory_path + "/posterior.png"
    plt.savefig(filename_posteriro_plot)
    plt.close()
    
    ###############################################TODO################################
    # Plot and save the file for each parameter
    ###################################################################################
    for i in range(len(test_list)):
        plt.figure(figsize=(8,10))
        az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        var_names=['mu_' +str(i+1)],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
        )
        filename_mu = directory_path + "/mu_"+str(i+1)+".png"
        plt.savefig(filename_mu)
        plt.close()
    
    ###########################################################################################
    ######################### Find the MAP value ##############################################
    ###########################################################################################
    MAP_sample_index_trace = torch.argmax(torch.tensor(log_posterior_vals))
    print("MAP_sample_index_trace\n", MAP_sample_index_trace)
    
    # MAP_sample_index, max_posterior_value, log_posterior_vals2 = compute_map(posterior_samples,geo_model_test,label_data,test_list,num_layers,directory_path, likelihood_std, dtype, device)
    # print("MAP_sample_index\n", MAP_sample_index)
    MAP_sample_index = MAP_sample_index_trace
    # print(torch.tensor(log_posterior_vals, dtype=dtype))
    # print(torch.tensor(log_posterior_vals2,dtype=dtype))
    # print(torch.tensor(log_posterior_vals,dtype=dtype) - torch.tensor(log_posterior_vals2, dtype=dtype))
    directory_path_MAP = directory_path +"/MAP"
    
    if not os.path.exists(directory_path_MAP):
        # Create the directory if it does not exist
        os.makedirs(directory_path_MAP)
        print(f"Directory '{directory_path_MAP}' was created.")
    else:
        print(f"Directory '{directory_path_MAP}' already exists.")
    
    ################################################################################
    #  Try Plot the data and save it as file in output folder
    ################################################################################

    RV_mu_post_MAP = {}
    
    for i in range(num_layers):
        RV_mu_post_MAP["mu_"+str(i+1)+"_post"] = posterior_samples["mu_"+str(i+1)][MAP_sample_index]
    
    # # Update the model with the new top layer's location
    interpolation_input = geo_model_test.interpolation_input
    print(RV_mu_post_MAP)
    counter2=1
    for interpolation_input_data in test_list[:num_layers]:
        interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])), RV_mu_post_MAP["mu_"+str(counter2)+"_post"])
        counter2=counter2+1

    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model_test.interpolation_options,
        data_descriptor=geo_model_test.input_data_descriptor,
        geophysics_input=geo_model_test.geophysics_input,
    )
    custom_grid_values_test = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    sp_coords_copy_test2 =interpolation_input.surface_points.sp_coords
    sp_cord= geo_model_test.transform.apply_inverse(sp_coords_copy_test2.detach().numpy())
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_final = df_sp_init.copy()
    df_sp_final["Z"] = sp_cord[:,2] 
    filename_final_sp = directory_path_MAP + "/Final_sp.csv"
    df_sp_final.to_csv(filename_final_sp)
    ################################################################################
    
    filename_posterior_model = directory_path_MAP + "/Posterior_model.png"
    
    geo_model_test_post = create_final_gempy_model_KSL_3_layer(refinement=7,filename=filename_posterior_model,sp_cord=sp_cord, save=False)
    
    
    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    gempy_label = np.round(custom_grid_values_post)
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(gempy_label):
        idx =gempy_label ==label_
        plt.scatter(z[idx],gempy_label[idx], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('label')
    plt.title("Dept vs label")
    filename = directory_path_MAP + "/gempy_model.png"
    plt.savefig(filename)
    plt.close()
    
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path_MAP+ "/Posterior_model.png"
    plt.savefig(filename_posterior_model)
    
    ###########################################################################################
    ######################### Find the mean value ##############################################
    ###########################################################################################
    directory_path_Mean = directory_path +"/Mean"
    # Check if the directory exists
    if not os.path.exists(directory_path_Mean):
        # Create the directory if it does not exist
        os.makedirs(directory_path_Mean)
        print(f"Directory '{directory_path_Mean}' was created.")
    else:
        print(f"Directory '{directory_path_Mean}' already exists.")
    
    RV_mu_post_Mean = {}
    
    for i in range(num_layers):
        RV_mu_post_Mean["mu_"+str(i+1)+"_post"] = posterior_samples["mu_"+str(i+1)].mean()
        RV_mu_post_Mean["mu_"+str(i+1)+"_std_post"] = posterior_samples["mu_"+str(i+1)].std()
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # # Update the model with the new top layer's location
    interpolation_input = geo_model_test.interpolation_input
    print(RV_mu_post_Mean)
    counter2=1
    for interpolation_input_data in test_list[:num_layers]:
        interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), RV_mu_post_Mean["mu_"+str(counter2)+"_post"])
        counter2=counter2+1

    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model_test.interpolation_options,
        data_descriptor=geo_model_test.input_data_descriptor,
        geophysics_input=geo_model_test.geophysics_input,
    )
    custom_grid_values_test = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    sp_coords_copy_test2 =interpolation_input.surface_points.sp_coords
    sp_cord= geo_model_test.transform.apply_inverse(sp_coords_copy_test2.detach().numpy())
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_final = df_sp_init.copy()
    df_sp_final["Z"] = sp_cord[:,2] 
    filename_final_sp = directory_path_Mean + "/Final_sp.csv"
    df_sp_final.to_csv(filename_final_sp)
    ################################################################################
    
   
    filename_posterior_model = directory_path_Mean + "/Posterior_model.png"
    geo_model_test_post = create_final_gempy_model_KSL_3_layer(refinement=7,filename=filename_posterior_model,sp_cord=sp_cord, save=False)
        
    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    gempy_label = np.round(custom_grid_values_post)
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(gempy_label):
        idx =gempy_label ==label_
        plt.scatter(z[idx],gempy_label[idx], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('label')
    plt.title("Dept vs label")
    filename = directory_path_Mean + "/gempy_model.png"
    plt.savefig(filename)
    plt.close()
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path_Mean+ "/Posterior_model.png"
    plt.savefig(filename_posterior_model)
    ###########################################################################################
    ######################### Find the mean + sigma ##############################################
    ###########################################################################################
    
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # # Update the model with the new top layer's location
    interpolation_input = geo_model_test.interpolation_input
    print(RV_mu_post_Mean)
    counter2=1
    for interpolation_input_data in test_list[:num_layers]:
        mean_value = RV_mu_post_Mean["mu_"+str(counter2)+"_post"] + RV_mu_post_Mean["mu_"+str(counter2)+"_std_post"]
        interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), mean_value )
        counter2=counter2+1

    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model_test.interpolation_options,
        data_descriptor=geo_model_test.input_data_descriptor,
        geophysics_input=geo_model_test.geophysics_input,
    )
    custom_grid_values_test = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    sp_coords_copy_test2 =interpolation_input.surface_points.sp_coords
    sp_cord= geo_model_test.transform.apply_inverse(sp_coords_copy_test2.detach().numpy())
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_final = df_sp_init.copy()
    df_sp_final["Z"] = sp_cord[:,2] 
    filename_final_sp = directory_path_Mean + "/Final_sp_mean_plus_std.csv"
    df_sp_final.to_csv(filename_final_sp)
    ################################################################################
    
    filename_posterior_model = directory_path_Mean + "/Posterior_model_mean_plus_std.png"
    geo_model_test_post = create_final_gempy_model_KSL_3_layer(refinement=7,filename=filename_posterior_model,sp_cord=sp_cord, save=False)
        
    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    gempy_label = np.round(custom_grid_values_post)
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(gempy_label):
        idx =gempy_label ==label_
        plt.scatter(z[idx],gempy_label[idx], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('label')
    plt.title("Dept vs label")
    filename = directory_path_Mean + "/gempy_model_mean_plus_std.png"
    plt.savefig(filename)
    plt.close()
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path_Mean+ "/Posterior_model_mean_plus_std.png"
    plt.savefig(filename_posterior_model)
    ###########################################################################################
    ######################### Find the mean - sigma ##############################################
    ###########################################################################################
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # # Update the model with the new top layer's location
    interpolation_input = geo_model_test.interpolation_input
    print(RV_mu_post_Mean)
    counter2=1
    for interpolation_input_data in test_list[:num_layers]:
        mean_value = RV_mu_post_Mean["mu_"+str(counter2)+"_post"] - RV_mu_post_Mean["mu_"+str(counter2)+"_std_post"]
        interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), mean_value )
        counter2=counter2+1

    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model_test.interpolation_options,
        data_descriptor=geo_model_test.input_data_descriptor,
        geophysics_input=geo_model_test.geophysics_input,
    )
    custom_grid_values_test = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    sp_coords_copy_test2 =interpolation_input.surface_points.sp_coords
    sp_cord= geo_model_test.transform.apply_inverse(sp_coords_copy_test2.detach().numpy())
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_final = df_sp_init.copy()
    df_sp_final["Z"] = sp_cord[:,2] 
    filename_final_sp = directory_path_Mean + "/Final_sp_mean_minus_std.csv"
    df_sp_final.to_csv(filename_final_sp)
    
    
    filename_posterior_model = directory_path_Mean + "/Posterior_model_mean_minus_std.png"
    geo_model_test_post = create_final_gempy_model_KSL_3_layer(refinement=7,filename=filename_posterior_model,sp_cord=sp_cord, save=False)
        
    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    gempy_label = np.round(custom_grid_values_post)
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(gempy_label):
        idx =gempy_label ==label_
        plt.scatter(z[idx],gempy_label[idx], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Depth')
    plt.ylabel('label')
    plt.title("Dept vs label")
    filename = directory_path_Mean + "/gempy_model_mean_minus_std.png"
    plt.savefig(filename)
    plt.close()
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path_Mean+ "/Posterior_model_mean_minus_std.png"
    plt.savefig(filename_posterior_model)

