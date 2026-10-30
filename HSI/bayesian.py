"""
model_factory: given a dataset and the chosen method, builds the matching
Pyro model instance and its corresponding observation tensor.

This is the one place that knows "PRE_SEGMENTED needs a single label
column (possibly GMM-derived)" and "INTEGRATED_BAYESIAN needs the full
feature matrix" -- run_pre_segmented.py / run_integrated_bayesian.py
call this instead of duplicating that logic themselves.
"""

import contextlib
import io
from typing import Tuple

from Dataset_loader import Method
import torch
import pyro
from pyro.nn import PyroModule
from pyro.infer import Predictive, MCMC, NUTS
from pyro.infer.autoguide import init_to_mean
import matplotlib.pyplot as plt
import arviz as az
from gempy_probability.plot_posterior import default_red, default_blue
import gempy as gp
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor

def create_pyro_model(
    dataset_final,   # UserDataset (post-preprocessing)
    geo_model_init,  # already-built initial GemPy model
    random_variable,  # List[Dict] of interface-update specs
    config,          # RunConfig
    method,          # Method
    data_type,       # torch.dtype
    device,          # torch.device
    output_dir,      # Output directory
    pre_seg_labels = None, # pre segmented labels 
    
) :
#-> Tuple[PyroModule, torch.Tensor]:
    """
    Build the Pyro model instance and observation tensor matching `method`.

    `geo_model_init` isn't used directly to build the model itself (the
    model classes take `geo_model_test` as an argument to `model_test`,
    not at construction time) -- it's accepted here so this factory has
    everything relevant in one call, and so future model variants that
    DO need geometry info at construction time can use it without
    changing the call signature.

    Returns:
        (model, obs_data): a PyroModule instance and its matching
        observation tensor, ready to pass into `run_inversion`.

    Raises:
        ValueError: for an unrecognized `method`.
    """
    
    
    num_layers = len(random_variable)
    
    if method ==Method.PRE_SEGMENTED:  # noqa: F405
        from pre_segmented import MyModel_PS
        model = MyModel_PS()
        if dataset_final.data.shape[1]==3:
            obs_data = dataset_final.labels
        if dataset_final.data.shape[1]>3:
            obs_data = pre_seg_labels
        
    elif method == Method.INTEGRATED_BAYESIAN:  # noqa: F405
        from integrated_bayesian import MyModel_IB
        model = MyModel_IB()
        obs_data = dataset_final.data[:,3:]
            
    else:
            raise ValueError(f"Unknown method: {method}")
    filename = output_dir/ "model_graph.png"
    pyro.clear_param_store()
    dot = pyro.render_model(model.posterior, model_args=(obs_data,random_variable,geo_model_init,num_layers, config, data_type, device),render_distributions=True, filename=filename)
    
    ###########
    ################################################################################
    # Prior
    ################################################################################
  
    prior = Predictive(model.posterior, num_samples=config.prior_number_samples)(obs_data,random_variable,geo_model_init,num_layers, config, data_type, device)
    
    # Create sub-dictionary without the avoid_key
    prior = dict((key, value) for key, value in prior.items() if "<" not in key and ">" not in key and key != "log_likelihood")
    plt.figure(figsize=(8,10))
    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    filename_prior_plot =  output_dir /"prior.png"
    plt.savefig(filename_prior_plot)
    plt.close()
    
    ################################################################################
    # Posterior 
    ################################################################################
    
    
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(model.posterior, step_size=config.step_size, adapt_step_size=config.adapt_step_size, target_accept_prob=config.target_accept_prob, max_tree_depth=config.max_tree_depth, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=config.posterior_number_samples,mp_context=config.mp_context, warmup_steps=config.posterior_warmup_steps,num_chains=config.posterior_num_chain, disable_validation=False)
    mcmc.run(obs_data,random_variable,geo_model_init,num_layers, config, data_type, device)
    
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
        trace = poutine.trace(lambda: model.posterior(obs_data,random_variable,geo_model_init,num_layers, config, data_type, device)).get_trace()
        
        # Apply the sampled parameter values to the model trace
        for name, value in sample_dict.items():
            trace.nodes[name]["value"] = value  # Set the sampled values in the trace
        
        # we need to re run our model to update the value of node which are dependent on posterior sample
        replayed_model = poutine.replay(model.posterior, trace=trace)
        trace = poutine.trace(replayed_model).get_trace(obs_data,random_variable,geo_model_init,num_layers, config, data_type, device)
        # Calculate the log joint (which includes log likelihood and log prior)
        log_joint = trace.log_prob_sum()
        
        log_posterior_vals.append(log_joint.item())  # .item() to extract scalar from tensor
    
    ###########################################################################################
    ######################### Find the MAP value ##############################################
    ###########################################################################################
    MAP_sample_index_trace = torch.argmax(torch.tensor(log_posterior_vals))
    print("MAP_sample_index_trace\n", MAP_sample_index_trace)
    
    
    
    print("MCMC summary results")
    
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        mcmc.summary()  
        summary_output = buf.getvalue()

    
    print(summary_output)
    
    with open(output_dir/'mcmc_summary_p.txt', 'w') as f:
                f.write(summary_output)
                
    
    posterior_samples_chain = {key: value[0] for key, value in posterior_samples_.items()}
    #print(posterior_samples_chain)
    posterior_predictive = Predictive(model.posterior, posterior_samples )(obs_data, random_variable, geo_model_init, num_layers, config, data_type, device)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    filename_posteriro_plot =  output_dir/"posterior.png"
    plt.savefig(filename_posteriro_plot)
    plt.close()
    
    

    
    results = {
        "prior": prior,
        "posterior_samples": posterior_samples,              # group_by_chain=False
        "posterior_samples_by_chain": posterior_samples_,    # group_by_chain=True
        "log_posterior_vals": log_posterior_vals,
        "MAP_sample_index": MAP_sample_index_trace,
        "num_layers": num_layers,
        "mcmc_summary_text": summary_output,
    }
    
    data.to_netcdf(output_dir/"bayesian_data.nc")
    torch.save(results, output_dir/"bayesian_results.pt")
    print(f"Saved Bayesian results to {output_dir / 'bayesian_results.pt'}")
    
    return results

       
    