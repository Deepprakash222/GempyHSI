"""
Dataset loading utilities for the cokriging + GMM geological inversion pipeline.

Supports pre-defined datasets (Salinas-A, KSL core data, synthetic label
models) as well as user-supplied data passed directly into a UserDataset
instance.

NOTE: `hsi_label`, `synthetic`, and `hsi_full` are imported with wildcard
imports below because that's how the original script had them. Once you
know exactly which names you use from each, replace these with explicit
imports (e.g. `from hsi_label import run_hsi_label`) -- wildcard imports
make it hard to trace where a given name comes from and can silently
shadow other names.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from typing import Callable, List, Optional, Union

import joblib 
import numpy as np
import pandas as pd
import scipy.io
import torch

import gempy as gp

# TODO: replace with explicit imports once you know what you use from each
# from hsi_label import *   # noqa: F401,F403
# from synthetic import *   # noqa: F401,F403
# from hsi_full import *    # noqa: F401,F403

from run_config import RunConfig
from model_existing import *
from preprocessing import preprocess_dataset


    

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_DIR = Path("../Dataset")
SALINAS_DIR = DATASET_DIR / "HSI_Salinas"
KSL_DIR = DATASET_DIR / "Fw__Hyperspectral_datasets_from_the_KSL_cores"

# Salinas-A ground-truth relabeling: maps original label values (as they
# appear when iterating np.unique(SalinasA_gt)) to a fixed, ordered set of
# class indices used throughout the rest of the pipeline.
SALINAS_LABEL_REMAP = [0, 6, 1, 5, 4, 3, 2]
SALINAS_UNLABELLED_CODE = 7  # pixels with this code (after remap) are dropped

# KSL core: only keep samples below this depth (in the same units as "Z")
KSL_DEPTH_CUTOFF = -700
# Fixed spatial location assigned to every KSL sample (single borehole/core)
KSL_X_LOCATION = 300
KSL_Y_LOCATION = 0

# Synthetic "Syn_label" boundary depths (in the same units as z)
SYN_LABEL_BOUNDARY_1 = -845
SYN_LABEL_BOUNDARY_2 = -825
SYN_LABEL_Z_RANGE = (-900, -700)
SYN_LABEL_N_POINTS = 250
SYN_LABEL_X_LOCATION = 300
SYN_LABEL_Y_LOCATION = 0

data_type = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataNotFoundError(Exception):
    """Raised when the dataset name is not found in the available datasets."""


@dataclass
class UserDataset:
    """
    Holds a dataset (either pre-defined or user-supplied) for analysis.

    Attributes:
        name: Name of the dataset. If it matches one of the pre-defined
            dataset names, `load_dataset` will populate `data`/`labels`
            automatically; otherwise `data` (and optionally `labels`) must
            be supplied directly.
        data: The dataset itself (coordinates + features). Populated by
            `load_dataset` for pre-defined datasets.
        startval: Lower bound (inclusive) on the "X" column used to slice
            a spatial column out of the Salinas-A image. Only used for
            the "SalinasA" dataset.
        endval: Upper bound (inclusive) on the "X" column, see `startval`.
        labels: Optional ground-truth labels, if available.
    """

    name: str
    data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None
    # startval: Optional[int] = 18
    # endval: Optional[int] = 22
    labels: Optional[Union[pd.Series, np.ndarray, torch.Tensor]] = None


class Method(str, Enum):
    """
    The two inference methods described in the paper.

    PRE_SEGMENTED: assumes an initial (soft) segmentation is available --
        either directly, as with synthetic (X, Y, Z, label) data, or
        obtained by first running a GMM on the raw features. Only method
        available for coordinate-only datasets, since there is no spectral
        information to run INTEGRATED_BAYESIAN on.
    INTEGRATED_BAYESIAN: uses the full feature set directly, with the GMM
        acting as the likelihood within the Bayesian inference itself.
        Requires spectral/feature columns beyond (X, Y, Z).
    """

    PRE_SEGMENTED = "pre_segmented"
    INTEGRATED_BAYESIAN = "integrated_bayesian"


# --------------------------------------------------------------------------- #
# Individual dataset loaders
# --------------------------------------------------------------------------- #

def _load_salinas_a(user_data: UserDataset) -> None:
    """Load the Salinas-A hyperspectral benchmark and slice out one column."""
    salinas_a = np.array(scipy.io.loadmat(SALINAS_DIR / "SalinasA.mat")["salinasA"])
    salinas_a_corrected = np.array(
        scipy.io.loadmat(SALINAS_DIR / "SalinasA_corrected.mat")["salinasA_corrected"]
    )
    salinas_a_gt = np.array(scipy.io.loadmat(SALINAS_DIR / "SalinasA_gt.mat")["salinasA_gt"])

    # Remap ground-truth labels to a fixed, ordered set of class indices.
    for original_value, new_value in zip(np.unique(salinas_a_gt), SALINAS_LABEL_REMAP):
        salinas_a_gt[salinas_a_gt == original_value] = new_value
    salinas_a_gt = 7 - salinas_a_gt

    height, width = salinas_a_gt.shape
    n_spectral_bands = salinas_a_corrected.shape[2]

    # Vectorized construction of the (X, Y, Z, Label, spectral...) table,
    # replacing the original double for-loop.
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_col = xx.reshape(-1)
    y_col = np.zeros(height * width)
    z_col = -yy.reshape(-1)
    label_col = salinas_a_gt.reshape(-1)
    spectral_cols = salinas_a_corrected.reshape(height * width, n_spectral_bands)

    data_hsi = np.column_stack([x_col, y_col, z_col, label_col, spectral_cols])
    data_hsi = torch.tensor(data_hsi, dtype=data_type)

    column_names = ["X", "Y", "Z", "Label"] + [
        f"feature_{i + 1}" for i in range(n_spectral_bands)
    ]
    df_hsi = pd.DataFrame(data_hsi, columns=column_names)

    # Drop unlabelled pixels.
    df_labelled = df_hsi.loc[df_hsi["Label"] != SALINAS_UNLABELLED_CODE].copy()

    # NOTE: spectral normalization is intentionally NOT done here --
    # it's handled once, downstream, by preprocess_dataset() in
    # preprocessing.py. Doing it here too would silently double-apply
    # z-score standardization.
 
    # Slice out a single spatial column, treated as a proxy "core".


    df_slice = df_labelled.loc[
        #(df_labelled["X"] >= user_data.startval) & (df_labelled["X"] <= user_data.endval)
        (df_labelled["X"] >= 18) & (df_labelled["X"] <= 22)
    ]

    user_data.labels = torch.tensor(df_slice["Label"].to_numpy())
    user_data.data = torch.tensor(df_slice.drop(columns="Label").to_numpy())
    print("Shape of Dataset:", user_data.data.shape)


def _load_ksl_layer3(user_data: UserDataset) -> None:
    """Load KSL core data (mineral-category RGB composite maps)."""
    filepath = KSL_DIR / "CuSp131.pkl"
    with open(filepath, "rb") as f:
        raw = joblib.load(f)

    column_names = []
    blocks = []
    n_samples = None

    for key, values in raw.items():
        if key == "XYZ":
            column_names += ["Borehole_id", "X", "Y", "Z"]
            n_samples = values.shape[0]
            blocks.append(np.ones((n_samples, 1)))
            blocks.append(values)
        else:
            column_names += [f"{key}_R", f"{key}_G", f"{key}_B"]
            blocks.append(values)

    combined = np.hstack(blocks)
    combined = combined[np.argsort(-combined[:, 3])]  # sort by depth (Z)

    df_ksl = pd.DataFrame(combined, columns=column_names)
    df_ksl = df_ksl[df_ksl["Z"] <= KSL_DEPTH_CUTOFF].copy()
    df_ksl["X"] = KSL_X_LOCATION
    df_ksl["Y"] = KSL_Y_LOCATION

    # NOTE: spectral normalization is intentionally NOT done here --
    # it's handled once, downstream, by preprocess_dataset() in
    # preprocessing.py. Doing it here too would silently double-apply
    # z-score standardization.
    
    # Drop the borehole-id column, keep X, Y, Z, and RGB features.
    data_hsi = df_ksl.iloc[:, 1:]

    user_data.data = torch.tensor(data_hsi.to_numpy())
    print("Shape of Dataset:", user_data.data.shape)


def _load_synthetic_label(user_data: UserDataset, *, depth_shift: float = 0.0) -> None:
    """
    Build a synthetic 3-class label column along a single vertical line.

    Args:
        depth_shift: Constant offset applied to both class boundaries,
            used to generate a shifted variant of the base synthetic model.
    """
    boundary_1 = SYN_LABEL_BOUNDARY_1 + depth_shift
    boundary_2 = SYN_LABEL_BOUNDARY_2 + depth_shift

    z = np.linspace(*SYN_LABEL_Z_RANGE, SYN_LABEL_N_POINTS)
    labels = np.ones_like(z)
    labels[z <= boundary_1] = 3
    labels[(z > boundary_1) & (z <= boundary_2)] = 2

    user_data.labels = torch.tensor(labels, dtype=data_type)
    xyz = torch.tensor(
        [[SYN_LABEL_X_LOCATION, SYN_LABEL_Y_LOCATION, z_val] for z_val in z],
        dtype=data_type,
    )
    user_data.data = xyz


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #

# TODO: "KSL" and "Syn_label_shift_20_error" were listed as valid names in
# the original script but had no implementation. Add loaders for them here
# once their intended behaviour is defined -- until then, requesting them
# will raise DataNotFoundError instead of silently producing an empty
# dataset (which is what the original code did).
_DATASET_LOADERS: dict[str, Callable[[UserDataset], None]] = {
    "SalinasA": _load_salinas_a,
    "KSL_layer3": _load_ksl_layer3,
    "Syn_label": _load_synthetic_label,
    "Syn_label_shift": lambda ud: _load_synthetic_label(ud, depth_shift=50.0),
}


def load_dataset(user_data: UserDataset) -> None:
    """
    Populate `user_data.data` (and `user_data.labels`, where available)
    either from a pre-defined dataset, or by normalising user-supplied data
    into a torch.Tensor.

    Raises:
        DataNotFoundError: if `user_data.name` isn't a known pre-defined
            dataset and no data was supplied directly.
    """
    if user_data.name in _DATASET_LOADERS:
        print(f"Loading pre-defined dataset: {user_data.name}")
        _DATASET_LOADERS[user_data.name](user_data)
    elif user_data.data is None and user_data.labels is None:
        raise DataNotFoundError(
            f"Dataset '{user_data.name}' not found and no data provided."
        )

    print(f"Dataset Name: {user_data.name}")

    # Normalise whatever format `data`/`labels` are in into torch.Tensor.
    if isinstance(user_data.data, pd.DataFrame):
        print("Data provided as pandas DataFrame.")
        user_data.data = torch.tensor(user_data.data.values, dtype=data_type)
        if user_data.labels is not None:
            user_data.labels = torch.tensor(user_data.labels.values, dtype=data_type)
    elif isinstance(user_data.data, np.ndarray):
        print("Data provided as NumPy array.")
        user_data.data = torch.tensor(user_data.data, dtype=data_type)
        if user_data.labels is not None:
            user_data.labels = torch.tensor(user_data.labels, dtype=data_type)
    elif isinstance(user_data.data, torch.Tensor):
        print("Data provided as PyTorch tensor.")
    else:
        print("Unsupported or missing data type.")

    print("Labels are provided." if user_data.labels is not None else "Labels are not provided.")


def describe_dataset_type(user_data: UserDataset) -> str:
    """
    Classify the dataset based on its shape/labels, mirroring the original
    `run_files` behaviour, but returning the classification instead of only
    printing it.
    """
    n_features = user_data.data.shape[1]
    has_labels = user_data.labels is not None

    if n_features > 3 and has_labels:
        kind = "Salinas type dataset"
    elif n_features > 3 and not has_labels:
        kind = "KSL type dataset"
    elif n_features == 3 and has_labels:
        kind = "synthetic type dataset"
    else:
        kind = "unrecognised dataset shape"

    print(kind)
    return kind


def available_methods(user_data: UserDataset) -> List[Method]:
    """
    Return the list of inference methods that are valid for this dataset.

    This depends ONLY on whether spectral/feature columns beyond (X, Y, Z)
    are present -- NOT on whether ground-truth labels are available.
    Labels (when present, e.g. for "SalinasA" or synthetic datasets) are
    used for post-hoc evaluation (accuracy, boundary error, etc.) and do
    not restrict or force a particular method. For datasets without labels
    (e.g. "KSL_layer3"), both methods are still valid to run -- only the
    evaluation approach afterward differs (falling back to unsupervised /
    internal validity metrics instead of ground-truth comparison).
    """
    n_features = user_data.data.shape[1]
    if n_features == 3:
        return [Method.PRE_SEGMENTED]
    return [Method.PRE_SEGMENTED, Method.INTEGRATED_BAYESIAN]


# --------------------------------------------------------------------------- #
# Dataset selection
# --------------------------------------------------------------------------- #

def select_dataset(
    name: str,
    data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    labels: Optional[Union[pd.Series, np.ndarray, torch.Tensor]] = None,
    # startval: int = 18,
    # endval: int = 22,
) -> UserDataset:
    """
    Select and load a dataset.

    If `name` matches one of the pre-defined datasets (see
    `_DATASET_LOADERS`), it is loaded automatically and `data`/`labels`
    are ignored. Otherwise, `data` (and optionally `labels`) must be
    supplied directly -- `name` is then just used as a label for this
    custom dataset.

    Args:
        name: Name of a pre-defined dataset, or an arbitrary identifier
            for a custom dataset supplied via `data`/`labels`.
        data: Required if `name` is not a pre-defined dataset.
        labels: Optional ground-truth labels for a custom dataset.
        startval, endval: Only used for the "SalinasA" dataset, to select
            which spatial column to slice out as a proxy core.

    Raises:
        DataNotFoundError: if `name` isn't pre-defined and `data` is None.
    """
    if name not in _DATASET_LOADERS and data is None:
        raise DataNotFoundError(
            f"'{name}' is not a pre-defined dataset ({list(_DATASET_LOADERS)}), "
            "so `data` must be supplied directly (with optional `labels`)."
        )

    user_data = UserDataset(
        name=name, data=data, labels=labels #, startval=startval, endval=endval
    )
    load_dataset(user_data)
    describe_dataset_type(user_data)
    return user_data


# --------------------------------------------------------------------------- #
# Method selection + pipeline execution
# --------------------------------------------------------------------------- #

# _METHOD_RUNNERS: dict = {
#     Method.PRE_SEGMENTED: run_hsi_label,          # noqa: F405
#     Method.INTEGRATED_BAYESIAN: run_hsi_full,     # noqa: F405
# }


"""
model_creation: automatically selects and calls the correct GemPy
initial-model builder for a given dataset, based on `dataset.name`.

get_geo_model_final_fn: looks up the matching "final" model builder
(callable, NOT called yet -- it needs posterior data first) for the
same dataset.

Add a new dataset by adding one entry to `_GEO_MODEL_REGISTRY` -- no
changes needed to `model_creation` or `get_geo_model_final_fn` themselves.
"""

from typing import Callable, Dict


# Each entry maps a dataset name to:
#   "init_fn":     builder for the initial model, called immediately.
#   "init_kwargs": extra kwargs specific to init_fn (e.g. depth_shift).
#   "final_fn":    builder for the final model, returned as a callable
#                  (not called here) -- used later once posterior data
#                  is available (see generate_final_model).
_GEO_MODEL_REGISTRY: Dict[str, dict] = {
    "Syn_label": {
        "init_fn": create_gempy_model_KSL_initial,  # noqa: F405
        "init_kwargs": {"depth_shift": (0.0, 0.0)},
        "final_fn": create_gempy_model_KSL_final,  # noqa: F405
    },
    "Syn_label_shift": {
        "init_fn": create_gempy_model_KSL_initial,  # noqa: F405
        "init_kwargs": {"depth_shift": (0.0, 0.0)},  # matches SYN_LABEL_BOUNDARY_1/2 + 50 shift
        "final_fn": create_gempy_model_KSL_final,  # noqa: F405
    },
    "KSL_layer3": {
        "init_fn": create_gempy_model_KSL_initial,  # noqa: F405
        "init_kwargs": {"depth_shift": (10.0, 30.0)},
        "final_fn": create_gempy_model_KSL_final,  # noqa: F405
    },
    "SalinasA": {
        "init_fn": create_gempy_model_Salinas_A_layer_initial,  # noqa: F405
        "init_kwargs": {},  # TODO: fill in once this function's signature is finalized
        "final_fn": create_gempy_model_Salinas_A_layer_final,  # noqa: F405 -- TODO: not yet written
    },
}


def _get_setup(dataset) -> dict:
    if dataset.name not in _GEO_MODEL_REGISTRY:
        raise ValueError(
            f"No GemPy model setup registered for dataset '{dataset.name}'. "
            f"Known datasets: {list(_GEO_MODEL_REGISTRY)}"
        )
    return _GEO_MODEL_REGISTRY[dataset.name]


def model_creation(
    dataset,  # UserDataset
    diretory
):
    """
    Build the initial GemPy model for `dataset`, automatically selecting
    the correct model-construction function (and any dataset-specific
    kwargs, e.g. `depth_shift`) based on `dataset.name`.

    Builds twice: once at refinement=7 (saved, for a nicer plot), then
    again at refinement=3 (not saved, faster -- used for the actual
    inference, since GemPy solutions at different refinements are
    independent). Prints the surface points table (physical coordinates)
    so random variables can be chosen by inspecting the printout.

    Returns:
        (geo_model_init, geo_model_final_fn):
          - geo_model_init: the built model (at refinement=3).
          - geo_model_final_fn: the callable used later to rebuild the
            "final" model from posterior data (see `generate_final_model`).
    """
    setup = _get_setup(dataset)
    init_fn = setup["init_fn"]
    init_kwargs = setup["init_kwargs"]
    gempy_mdodel_final = setup["final_fn"]

    filename= str(diretory / "initial_model.png")
    filename_initial_sp = str(diretory /"Initial_sp.csv")
    filename_initial_op = str(diretory /"Initial_op.csv")
    # Refinement=7 for a nicer plot (saved).
    geo_model_init = init_fn(refinement=7, filename=filename, save=True, **init_kwargs)
    # Refinement=3 for actual inference (faster; GemPy solutions at
    # different refinements are computed independently).
    geo_model_init = init_fn(refinement=3, filename=filename, save=False, **init_kwargs)

    normalised_coord = geo_model_init.interpolation_input.surface_points.sp_coords.copy()
    physical_coord = geo_model_init.transform.apply_inverse(normalised_coord)  
    
    df_sp_init = geo_model_init.surface_points.df
    df_or_init = geo_model_init.orientations.df
    print("#################################################")
    print("Gempy Physical coordinates")
    print("#################################################")
    print(df_sp_init)
    print("#################################################")
    df_sp_init.to_csv(filename_initial_sp)
    df_or_init.to_csv(filename_initial_op)
        
    return geo_model_init, physical_coord, normalised_coord, gempy_mdodel_final 

def run_bayesian(
    dataset, 
    geo_model_init ,
    geo_model_final,
    config,
    random_variable,
    method,
    output_dir,
    pre_seg_labels= None,
    ):
    
    from seeding import set_seed
    set_seed(config.seed)
 
    n_features = dataset.data.shape[1]
 
    if n_features == 3:
        if method != Method.PRE_SEGMENTED:  # noqa: F405
            print(
                f"Wrong method selected: dataset '{dataset.name}' has only {n_features} "
                f"columns (coordinates only), so only '{Method.PRE_SEGMENTED.value}' "  # noqa: F405
                f"is valid -- got '{method.value}'."
            )
            return
        if dataset.labels is None:
            print(
                f"Missing labels: dataset '{dataset.name}' has only coordinate columns "
                f"(no spectral features), so labels are required as the observation data "
                f"for '{Method.PRE_SEGMENTED.value}' -- but dataset.labels is None."  # noqa: F405
            )
            return
        
    elif n_features > 3:
        if method == Method.PRE_SEGMENTED:  # noqa: F405
            print("We can use the pre segmented label as the obsrvation data")
            
    else:
        print(
            f"Wrong dataset: '{dataset.name}' has {n_features} column(s), "
            f"which is fewer than the minimum required (3: X, Y, Z)."
        )
        return
 
    print(f"Dataset and method are compatible: '{dataset.name}' ({n_features} columns), method='{method.value}'.")
    
    print("##############################################################################\n")
    print("Running Bayesian inversion pipeline ...\n")
    print("##############################################################################\n")
    
    from bayesian import create_pyro_model
    
    bayesian_results = create_pyro_model(dataset_final=dataset,
                      random_variable=random_variable,
                      geo_model_init=geo_model_init,
                      config=config, 
                      method=method,
                      data_type=data_type,
                      device=device,
                      pre_seg_labels = pre_seg_labels,
                      output_dir=output_dir)
    return bayesian_results

def calculate_posterior_statistics(bayesian_results, 
                                   config,
                                   output_dir
                                   ):
    """
    Calculate posterior statistics (mean, std, quantiles) from the Bayesian results.

    Args:
        bayesian_results: The results returned from the Bayesian inference. 
        
    """
    # Assuming bayesian_results is a dictionary containing posterior samples
    
    prior = bayesian_results.get("prior", None)
    posterior_samples = bayesian_results.get("posterior_samples", None)
    posterior_samples_by_chain = bayesian_results.get("posterior_samples_by_chain", None)
    log_posterior_vals = bayesian_results.get("log_posterior_vals", None)
    num_layers  = bayesian_results.get("num_layers", None)
    MAP_sample_index = bayesian_results.get("MAP_sample_index", None)
    mcmc_summary_text = bayesian_results.get("mcmc_summary_text", None)
    
    
    if posterior_samples is None:
        print("No posterior samples found in the results.")
        return None
    
    from posterior_analysis import posterior_samples_to_dataframe, summarize_posterior_dataframe
    
    df = posterior_samples_to_dataframe(posterior_samples)
    
    
    print(MAP_sample_index)
    
    map_row = df.iloc[MAP_sample_index.item()]  # Get the MAP estimate row
     # e.g. mu_1: -845.02, mu_2: -824.97
    
    
    corr = df.corr()                       # correlation between mu_1, mu_2, ... (useful given your layer-ordering constraint)
    
    summary = summarize_posterior_dataframe(df)
    summary["MAP"] = map_row
    summary.to_csv( output_dir/"posterior_summary.csv")

    # --- Load ---
    
    import arviz as az
    from gempy_probability.plot_posterior import default_red, default_blue
    data = az.from_netcdf(output_dir/"bayesian_data.nc")   # matches whatever path you used in idata.to_netcdf(...)
    #print(data.sample_stats)   # confirm "diverging" is listed as a variable here
    
    ###############################################TODO################################
    # Divergane statistics: print the total number of divergences and the divergence rate
    ###################################################################################
    print('#' * 50)
    print("Divergence statistics:")
    print('#' * 50)
    divergences = data.sample_stats["diverging"].values   # 
    print("Total divergences:", divergences.sum())
    print("Rate: {:.2f}%".format(100 * divergences.mean()))
    print('#' * 50)
    ###############################################TODO################################
    # Plot and save the file for each parameter
    ###################################################################################
    for i in range(num_layers):
        var_name = f"mu_{i+1}"
        
        fig = plt.figure(figsize=(8, 10))
        ax = az.plot_density(
            data=[data.posterior, data.prior],
            shade=.9,
            bw=config.bandwith,
            var_names=[var_name],
            data_labels=["Posterior Predictive", "Prior Predictive"],
            colors=[default_red, default_blue],
        )
        target_ax = np.atleast_1d(ax).flatten()[0]
        target_ax.set_title(rf"$\mu_{{{i+1}}}$", fontsize=14)
        #target_ax.set_title(rf"$\mu_{{{2}}}$", fontsize=14)
        target_ax.set_axisbelow(True)
    
        filename_mu = output_dir / (var_name + ".png")
        plt.savefig(filename_mu)
        plt.close()
        
    statistics = {
        "summary": summary,
        "corr": corr
    }
    
    return statistics

def realization_to_gempy_model(statistics, 
                               geo_model_init, 
                               geo_model_final_fn, 
                               random_variable, 
                               custom_grid,
                               config,
                               output_dir):
    """
    Convert a single posterior realization into a GemPy model.

    Args:
        statistics: The posterior statistics calculated from the Bayesian results.
        geo_model_init: The initial GemPy model.
        geo_model_final_fn: The callable to create the final GemPy model.
        realization: A dictionary containing the posterior values for the random variables.
        random_variable: The list of random variable specifications.
        slope: The slope parameter for the GemPy model.

    Returns:
        A GemPy model corresponding to the given realization.
    """
    # Extract the posterior data from the realization based on the random_variable specification
    summary = statistics["summary"]
    posterior_mean = list(torch.tensor(summary["mean"].to_numpy(), dtype=data_type))
    posterior_mean_plus_std = list(torch.tensor((summary["mean"] + summary["std"]).to_numpy(), dtype=data_type))
    posterior_mean_minus_std = list(torch.tensor((summary["mean"] - summary["std"]).to_numpy(), dtype=data_type))
    posterior_MAP = list(torch.tensor(summary["MAP"].to_numpy(), dtype=data_type))
    
    posterior_data = {"posterior_mean": posterior_mean,
                      "posterior_mean_plus_std": posterior_mean_plus_std,
                      "posterior_mean_minus_std": posterior_mean_minus_std,
                      "posterior_MAP": posterior_MAP}
    output_segmentation = {}
    for realization , value in posterior_data.items():
        
        geo_model_realization = generate_final_model(
                geo_model=geo_model_init,
                geo_model_final_fn=geo_model_final_fn,
                random_variable=random_variable,
                num_layers=len(random_variable),
                posterior_data=value,
                slope=config.slope_gempy,
                filename=output_dir/realization,  # No need to save an image for each realization
                save=True,
            )
        gp.set_custom_grid(geo_model_realization.grid, xyz_coord=custom_grid)
        geo_model_realization.interpolation_options.mesh_extraction = False
        geo_model_realization.interpolation_options.sigmoid_slope = config.slope_gempy
        sol=gp.compute_model(geo_model_realization)
        # Compute and observe the thickness of the geological layer 
        model_output = torch.tensor(geo_model_realization.solutions.octrees_output[0].last_output_center.custom_grid_values, dtype=data_type, device=device)
        output_segmentation[realization] = model_output
    return output_segmentation
    

def main() -> None:
    
    config = RunConfig.from_yaml("config.yaml")
    
    geo_model_init = None
    geo_model_final = None

    # Step 1: load the dataset (pre-defined or user-supplied)
    print("#" * 50)
    print("Loading dataset ....")
    print("#" * 50)

    
    #dataset = select_dataset(name="Syn_label")
    #dataset = select_dataset(name="Syn_label_shift")
    #dataset = select_dataset(name="SalinasA")
    dataset = select_dataset(name="KSL_layer3")
    
    print("#" * 50)
    print("Dataset Loaded Successfully ....")
    print("#" * 50)
    print(dataset.data.shape)
    
    # step 2 : preprocess the dataset (Normalization and PCA if needed)
    print("#" * 50)
    print("Preprocessing dataset ....")
    print("#" * 50)
    
    dataset_final = preprocess_dataset(dataset)
    
    initial_dir = config.initial_output_dir(dataset_final.name)

    # step 3 : Initialize the GemPy model based on the dataset
    print("#" * 50)
    print("Initializing models ....")
    print("#" * 50)
    geo_model_init, physical_coord, normalised_coord, geo_model_final = model_creation(dataset, initial_dir)
    
    
    normalised_coord_t = torch.as_tensor(normalised_coord, dtype=data_type)
    
    print("Gempy Transformed coordinates")
    print("# " * 50)
    print(normalised_coord_t)
    print("# " * 50)
    

    # Step 4: Choose the data point to be Random variable and it's prior distribution
    
    random_variable = [
        {
            "update":"interface_data",
            "id":torch.tensor([1]), 
            "direction":"Z", 
            "prior_distribution":"normal",
            "normal":{"mean":torch.tensor(normalised_coord[1,2],dtype=data_type, device=device), "std":torch.tensor(0.02,dtype=data_type, device=device)}
        },
        {
            "update":"interface_data",
            "id":torch.tensor([4]), 
            "direction":"Z", 
            "prior_distribution":"normal",
            "normal":{"mean":torch.tensor(normalised_coord[4,2],dtype=data_type, device=device), "std":torch.tensor(0.02,dtype=data_type, device=device)}
        },
        ]
    
    # random_variable = [
    #     {
    #         "update":"interface_data",
    #         "id":torch.tensor([1]),
    #         "direction":"Z",
    #         "prior_distribution":"normal",
    #         "normal":{"mean":torch.tensor(normalised_coord[1,2],dtype=data_type, device=device), "std":torch.tensor(0.3,dtype=data_type, device=device)}
    #     },
    #     {
    #         "update":"interface_data",
    #         "id":torch.tensor([4]),
    #         "direction":"Z",
    #         "prior_distribution":"normal",
    #         "normal":{"mean":torch.tensor(normalised_coord[4,2],dtype=data_type, device=device), "std":torch.tensor(0.3,dtype=data_type, device=device)}
    #     },
    #     {
    #         "update":"interface_data",
    #         "id":torch.tensor([7]),
    #         "direction":"Z",
    #         "prior_distribution":"normal",
    #         "normal":{"mean":torch.tensor(normalised_coord[7,2],dtype=data_type, device=device), "std":torch.tensor(0.3,dtype=data_type, device=device)}
    #     },
    #     {
    #         "update":"interface_data",
    #         "id":torch.tensor([12]),
    #         "direction":"Z",
    #         "prior_distribution":"normal",
    #         "normal":{"mean":torch.tensor(normalised_coord[12,2],dtype=data_type, device=device), "std":torch.tensor(0.3,dtype=data_type, device=device)}
    #     },
    # ]
    
   
    #step 5: chose the custom grid location, where data is observed 

    custom_grid = (dataset.data[:,0:3]).numpy()
    print("#" * 50)
    print("Active grids before setting custom grid:\n", geo_model_init.grid.active_grids)
    print("Setting custom grid...")
    gp.set_custom_grid(geo_model_init.grid, xyz_coord=custom_grid)
    
    geo_model_init.interpolation_options.mesh_extraction = False
    
    geo_model_init.interpolation_options.sigmoid_slope = config.slope_gempy 
    sol= gp.compute_model(geo_model_init)
    # Compute and observe the thickness of the geological layer 
    model_output = torch.tensor(geo_model_init.solutions.octrees_output[0].last_output_center.custom_grid_values, dtype=data_type, device=device)
    print("#" * 50)
    # step 6: Initial performane of the model
    print("Calculating initial performance of the model ....")
    from performance_matrix import calculate_matrix
    matrix, label_data = calculate_matrix(dataset= dataset_final, 
                     model_output=model_output,
                     n_clusters=config.clusters, 
                     seed=config.seed,
                     classical_method=True)
    print("#" * 50)
    # Change the backend to PyTorch for probabilistic modeling
    
    from gempy_engine.core.backend_tensor import BackendTensor
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    geo_model_init.interpolation_options.sigmoid_slope = config.slope_gempy
    
    # Step 8: Run the pipeline by choosing the alorithm and the dataset
    print("#" * 50)
    method = Method.PRE_SEGMENTED
    #method = Method.INTEGRATED_BAYESIAN
    print("Selected method:", method.value)
    
    run_dir = config.run_output_dir(dataset_final.name, method.value)
    
    pre_seg_labels = None
    
    if dataset_final.data.shape[1] > 3 and method == Method.PRE_SEGMENTED.value:
        from performance_matrix import cluster_acc
        
        pre_seg_labels= label_data["GMM"]
        
        _, label_order, _, _ = cluster_acc( pre_seg_labels, torch.round(model_output))   
        # reaarange the label information so it is would be consistent with ground truth label
        pre_seg_labels = torch.tensor([label_order[x-1] +1  for x in pre_seg_labels], dtype=data_type, device=device)
        
    # Step 9: Evaluate the performance of the final model and compare with the initial model
    import os
    results_path = run_dir/"bayesian_results.pt"
    print("#" * 50)
    if config.rerun_bayesian or not os.path.exists(results_path):
        print("Running Bayesian inversion...")
        bayesian_results = run_bayesian(dataset = dataset_final, 
                    geo_model_init=geo_model_init, 
                    geo_model_final=geo_model_final, 
                    config=config, 
                    random_variable=random_variable, 
                    method=method,
                    pre_seg_labels = pre_seg_labels,
                    output_dir = run_dir)
    else:
        print(f"Loading saved Bayesian results from {results_path}")
        bayesian_results = torch.load(results_path, weights_only=False)
        
    
    print("#" * 50)
    print("Calculating posterior statistics...")
    statistics = calculate_posterior_statistics(bayesian_results, config, run_dir)
    
    output_segmentation = realization_to_gempy_model(statistics =statistics, 
                            geo_model_init = geo_model_init, 
                            geo_model_final_fn = geo_model_final, 
                            random_variable = random_variable, 
                            custom_grid = custom_grid,
                            config = config,
                            output_dir = run_dir)
    for key, values in output_segmentation.items():
        print(key)

        matrix_data, _ = calculate_matrix(
            dataset=dataset_final,
            model_output=values,
            n_clusters=config.clusters,
            seed=config.seed,
            classical_method=False
        )

        matrix.update({
            key_output.removesuffix("_model") + f"_{key}": value_output
            for key_output, value_output in matrix_data.items()
            if key_output.endswith("_model") or key_output.endswith(" model")
        })
        
        #matrix[key] = matrix_data["Overall accuracy (model)"]
    
    df = (pd.DataFrame([matrix]).T).round(2)
    df.index.name = "Name of method"
    df.columns = ["accuracy"]
    df.to_csv(run_dir/"matrix.csv")
    
if __name__ == "__main__":
    print("#" * 50)
    print("Script started...")
    print("#" * 50)
    
    start_time = datetime.now()

    main()

    end_time = datetime.now()
    print("Script ended...")
    print(f"Elapsed time: {end_time - start_time}")
    
    print("#" * 50)