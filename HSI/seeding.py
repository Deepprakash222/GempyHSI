"""
Single shared seeding utility, called once (from run_bayesian) rather
than duplicated across each inversion method's file.
"""
 
import numpy as np
import torch
 
 
def set_seed(seed: int = 42) -> None:
    """
    Seed numpy, torch, and pyro, and configure torch/multiprocessing for
    deterministic, reproducible behaviour.
    """
    import pyro
 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
 
    # # Safe to call even if already set; guards against re-invocation
    # # within the same process raising a RuntimeError.
    # try:
    #     torch.multiprocessing.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")