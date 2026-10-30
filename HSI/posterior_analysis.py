"""
Convert a posterior_samples dict ({"mu_1": tensor([...]), ...}) into a
pandas DataFrame for convenient statistics, quantiles, correlations, and
CSV export.
"""

import numpy as np
import pandas as pd
import torch


def posterior_samples_to_dataframe(posterior_samples: dict) -> pd.DataFrame:
    """
    Convert a dict of {name: 1D array-like} into a DataFrame, one column
    per variable, one row per posterior draw.

    Handles torch.Tensor values (including ones still on GPU / requiring
    grad) by converting to plain numpy first.
    """
    columns = {}
    for name, values in posterior_samples.items():
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        else:
            values = np.asarray(values)
        columns[name] = values

    return pd.DataFrame(columns)


def summarize_posterior_dataframe(
    df: pd.DataFrame,
    credible_interval: float = 0.90,
) -> pd.DataFrame:
    """
    Summary statistics per variable: mean, std, a credible interval (via
    quantiles)
 
    Args:
        df: Output of `posterior_samples_to_dataframe`.
        credible_interval: e.g. 0.90 (default) for a 90% credible
            interval 5th/95th percentiles.
        
    """
    alpha = (1 - credible_interval) / 2
    summary = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        f"q{alpha*100:.1f}%": df.quantile(alpha),
        f"q{(1-alpha)*100:.1f}%": df.quantile(1 - alpha),
    })
    return summary