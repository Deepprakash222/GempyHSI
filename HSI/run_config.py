"""
RunConfig: all tunable parameters for a single run, in one place, loadable
from (and saveable to) a YAML file -- so a user can hand-edit a config
file instead of touching code, and every run can be archived alongside
the config that produced it (useful for reproducibility in the paper).
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml  # pip install pyyaml

# Which RunConfig fields actually affect each method's behavior -> which
# ones belong in that method's output-directory name. Shared params
# (warmup, samples, chains, slope, enforce_ordering) apply to both; the
# likelihood-specific ones (scale vs likelihood_std) don't -- keeping
# them separate means two PRE_SEGMENTED runs that only differ in `scale`
# (which PRE_SEGMENTED never even uses) don't get different folders for
# no reason.
_METHOD_PARAM_FIELDS = {
    "pre_segmented": [
        "posterior_warmup_steps",
        "posterior_number_samples",
        "posterior_num_chain",
        "slope_gempy",
        "likelihood_std",
        "enforce_ordering",
    ],
    "integrated_bayesian": [
        "posterior_warmup_steps",
        "posterior_number_samples",
        "posterior_num_chain",
        "slope_gempy",
        "scale",
        "enforce_ordering",
    ],
}

# Short names used in the folder string instead of the full field name.
_PARAM_ABBREV = {
    "posterior_warmup_steps": "warmup_",
    "posterior_number_samples": "samples_",
    "posterior_num_chain": "chains_",
    "slope_gempy": "slope_",
    "scale": "scale_",
    "likelihood_std": "std_",
    "enforce_ordering": "order_",
}


def validate_method(method_name: str) -> None:
    """Check `method_name` is one _METHOD_PARAM_FIELDS knows about --
    call this wherever a method value first enters the pipeline."""
    if method_name not in _METHOD_PARAM_FIELDS:
        raise ValueError(f"Unknown method {method_name!r}; expected one of {sorted(_METHOD_PARAM_FIELDS)}")

@dataclass
class RunConfig:
    seed: int = 42
    clusters: int = 6 
    directory_path: str = "./Results"
    slope_gempy: float = 45.0
    scale: float = 10.0
    # ^ Softmax "slope"/temperature used to convert GemPy's continuous
    # custom_grid_values into soft one-hot class assignments (gamma_nk)
    # for the Integrated Bayesian likelihood: gamma_nk = softmax(scale * distances).
    # Larger scale -> sharper, more confident (near-one-hot) assignments;
    # smaller scale -> softer, more uncertain assignments. A natural
    # target for the same kind of sensitivity sweep already done for
    # slope_gempy and likelihood_std.
    likelihood_std: float = 0.4
    enforce_ordering: bool = True
    # ^ Whether to apply the mu_1 > mu_2 > ... monotonic layer-ordering
    # constraint during inference (see sample_interface_priors_and_update
    # / _enforce_layer_ordering in prior.py). Set to False to skip it --
    # e.g. to compare inversion behaviour with and without the
    # constraint, or if ordering is already guaranteed some other way.
    rerun_bayesian: bool = True
    # ^ If True (default), always run the (expensive) Bayesian inversion.
    # If False, load previously-saved results from
    # {run_output_dir()}/bayesian_results.pt instead of re-running MCMC --
    # useful for iterating on postprocessing/plotting without waiting for
    # inference again. Falls back to running inversion anyway if no saved
    # results file exists yet, regardless of this setting.
    
     # --- NUTS/MCMC sampler parameters (all of Predictive/MCMC's tunables, together) ---
    prior_number_samples: int = 1000
    posterior_number_samples: int = 250
    posterior_warmup_steps: int = 250
    posterior_num_chain: int = 5
    step_size: float = 0.0085
    adapt_step_size: bool = True
    target_accept_prob: float = 0.75
    max_tree_depth: int = 10
    mp_context: str = "spawn"
    bandwidth: float = 0.09
    
    def initial_output_dir(self, dataset_name: str) -> Path:
        """
        Results/{dataset_name}/ -- outputs shared across every method run
        on this dataset: the initial (pre-inversion) performance matrix,
        GMM/K-means baselines, pseudo-labels, etc. Created if missing.
        """
        path = Path(self.directory_path) / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_output_dir(self, dataset_name: str, method_name: str) -> Path:
        """
        Results/{dataset_name}/{method_name}/{param_string}/ -- outputs
        specific to one (dataset, method, hyperparameter) combination.
        `param_string` only encodes the parameters that actually affect
        *this* method (see _METHOD_PARAM_FIELDS below), so a
        PRE_SEGMENTED run is named by likelihood_std (not scale) and an
        INTEGRATED_BAYESIAN run by scale (not likelihood_std). Created
        if missing.
        """
        try:
            field_names = _METHOD_PARAM_FIELDS[method_name]
        except KeyError:
            raise ValueError(
                f"Unknown method {method_name!r}; expected one of {sorted(_METHOD_PARAM_FIELDS)}"
            )

        parts = []
        for field_name in field_names:
            value = getattr(self, field_name)
            abbrev = _PARAM_ABBREV[field_name]
            parts.append(f"{abbrev}{value:g}" if isinstance(value, float) else f"{abbrev}{value}")
        param_string = "_".join(parts)

        path = Path(self.directory_path) / dataset_name / method_name / param_string
        path.mkdir(parents=True, exist_ok=True)
        return path


    @classmethod
    def from_yaml(cls, path: "str | Path") -> "RunConfig":
        """Load a RunConfig from a YAML file. Unknown keys raise an error
        (rather than being silently ignored) to catch config typos early."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        known_fields = {f.name for f in fields(cls)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")

        return cls(**data)

    def to_yaml(self, path: "str | Path") -> None:
        """Save this RunConfig to a YAML file, e.g. alongside a run's
        outputs, so the exact parameters used are archived."""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def build_config_from_cli() -> RunConfig:
    """
    Build a RunConfig from command-line arguments -- the *only* place
    argparse should be used; everything downstream takes a RunConfig
    object, not argv, so it also works when called programmatically
    (notebook, tests, run_pipeline) without touching sys.argv at all.

    Also accepts `--config path/to/file.yaml` to load a full config from
    a file instead of (or as a base for) individual CLI flags.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run configuration")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a YAML config file. If given, other flags are ignored.")
    for f in fields(RunConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=None)
    args = parser.parse_args()

    if args.config is not None:
        return RunConfig.from_yaml(args.config)

    overrides = {f.name: getattr(args, f.name) for f in fields(RunConfig)
                 if getattr(args, f.name) is not None}
    return RunConfig(**overrides)