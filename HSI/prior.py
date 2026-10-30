"""
Shared Pyro-model building blocks, reused by both the Pre-Segmented and
Integrated-Bayesian models: sampling each declared interface's prior,
updating GemPy's surface points accordingly, and enforcing a monotonic
layer-ordering constraint. Only the likelihood differs between the two
methods -- this file has nothing likelihood-specific in it.
"""
import os
import torch
import pyro
import gempy_engine

import pyro.distributions as dist

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def sample_interface_priors_and_update(
    random_variable: list,
    geo_model_test,
    num_layers: int,
    dtype: torch.dtype,
    device: torch.device,
    enforce_ordering: bool = True,

) :
    """
    Sample each declared random variable (mu_1, mu_2, ...) from its prior,
    update `geo_model_test`'s surface points in place accordingly, and
    enforce a monotonic layer-ordering constraint between adjacent
    interfaces.

    Args:
        random_variable: List of interface-update specs -- the same
            list passed throughout the rest of the pipeline (e.g. to
            `run_bayesian`, `generate_final_model`), describing which
            surface points are treated as random variables and their
            priors.
        geo_model_test: GemPy model whose surface points get updated.
        num_layers: Number of entries in `random_variable` to process.
        dtype, device: Used for the ordering-constraint Delta distribution.

    Returns:
        Dict mapping "mu_i" -> sampled value, for i = 1..num_layers.
        (Named `mu_samples`, not `random_variable`, to avoid clashing
        with the `random_variable` list of specs above.)

    Raises:
        ValueError: on an unsupported prior_distribution or direction.
    """
    
    # print(
    #     "PID:",
    #     os.getpid(),
    #     "Model ID:",
    #     id(geo_model_test),
    #     "Interpolation ID:",
    #     id(geo_model_test.interpolation_input),
    # )
    mu_samples = {}
    interpolation_input = geo_model_test.interpolation_input
    initial_interpolation_input = interpolation_input
    
    for counter, spec in enumerate(random_variable[:num_layers], start=1):
        if spec["update"] != "interface_data":
            continue

        name = f"mu_{counter}"

        if spec["prior_distribution"] == "normal":
            mean = spec["normal"]["mean"]
            std = spec["normal"]["std"]
            mu_samples[name] = pyro.sample(name, dist.Normal(mean, std))
        elif spec["prior_distribution"] == "uniform":
            low = spec["uniform"]["min"]
            high = spec["uniform"]["max"]  # NOTE: was buggy in the original ("min" used twice)
            mu_samples[name] = pyro.sample(name, dist.Uniform(low, high))
        else:
            raise ValueError(f"Unsupported prior_distribution: {spec['prior_distribution']!r}")

        direction = spec["direction"]
        if direction not in _AXIS_INDEX:
            raise ValueError(f"Unknown direction {direction!r} (expected 'X', 'Y', or 'Z').")
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([spec["id"]]), torch.tensor([_AXIS_INDEX[direction]])),
            mu_samples[name],
        )
        
    
    if enforce_ordering:
        _enforce_layer_ordering(mu_samples, random_variable, dtype, device)
        
    after_update_interpolation_input  = interpolation_input
    #interpolation_input = initial_interpolation_input
    #print("after the update\n", interpolation_input.surface_points.sp_coords)
    coords = (
        interpolation_input
        .surface_points
        .sp_coords
    )

    
    geo_model_test.solutions = gempy_engine.compute_model(
                    interpolation_input=interpolation_input,
                    options=geo_model_test.interpolation_options,
                    data_descriptor=geo_model_test.input_data_descriptor,
                    geophysics_input=geo_model_test.geophysics_input,
                )
                
    #print("after the update\n", geo_model_test.interpolation_input.surface_points.sp_coords)
    
    output_data = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    #print("Output finite:", torch.isfinite(output_data).all())
    #print("Output:", output_data)
# --- Diagnostic: check for NaN and print everything relevant ---
    if torch.isnan(output_data).any() :
            print("=" * 60)
            print("custom grid:", output_data)
            print("Coordinates finite:", torch.isfinite(coords).all())
            print("Coordinates:", coords)
            print("Active grids before setting custom grid:\n", geo_model_test.grid.active_grids)
            print("initial interpolation_input\n",initial_interpolation_input.surface_points.sp_coords)
            print("after interpolation_input\n",interpolation_input.surface_points.sp_coords)
            print("=" * 60)
            exit()
    
    return output_data


def _enforce_layer_ordering(
    mu_samples: dict,
    random_variable: list,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """
    Enforce mu_1 > mu_2 > ... > mu_n (monotonic layer ordering), plus loose
    bounds on the first and last interface relative to their own prior
    mean +/- 2 std.

    Implemented via `pyro.sample(..., obs=<bool condition>)` against a
    Delta(1.0) distribution -- a standard trick for injecting a hard
    constraint: the observed value (True, cast to 1.0) has probability 1
    under Delta(1.0) if satisfied, and probability 0 (log-prob -inf) if
    violated, effectively ruling out configurations that break the
    ordering.

    Args:
        mu_samples: Dict of sampled values, as returned by
            `sample_interface_priors_and_update`.
        random_variable: The same list of interface-update specs.
    """
    one = torch.tensor(1.0, dtype=dtype, device=device)
    n = len(random_variable)

    for i in range( n + 1):
        if i == 0:
            bound = random_variable[0]["normal"]["mean"] + 2 * random_variable[0]["normal"]["std"]
            pyro.sample(
                f"mu_{i+1} < mu_{i+1} + 2*std", dist.Delta(one),
                obs=(mu_samples[f"mu_{i+1}"] < bound),
            )
        elif i == n:
            bound = random_variable[-1]["normal"]["mean"] - 2 * random_variable[-1]["normal"]["std"]
            pyro.sample(
                f"mu_{i} > mu_{i} - 2*std", dist.Delta(one),
                obs=(mu_samples[f"mu_{i}"] > bound),
            )
        else:
            pyro.sample(
                f"mu_{i} > mu_{i+1}", dist.Delta(one),
                obs=(mu_samples[f"mu_{i}"] > mu_samples[f"mu_{i+1}"]),
            )