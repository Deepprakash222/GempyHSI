"""
Optional, composable preprocessing steps applied to a UserDataset after
loading and before model setup. Each step is a plain function operating
on a UserDataset and returning a NEW UserDataset (via dataclasses.replace)
rather than mutating the one passed in.

"""

from dataclasses import replace
from typing import Callable, List, Optional

import numpy as np
import torch
from scipy.stats import zscore
from sklearn.decomposition import PCA

PreprocessFn = Callable[["UserDataset"], "UserDataset"]

# Assumes the first 3 columns of `data` are always (X, Y, Z); everything
# after that is feature/spectral columns.
_N_COORD_COLS = 3


def standardize(dataset: "UserDataset") -> "UserDataset":
    """Z-score standardize each row's feature columns (columns beyond X,Y,Z)."""
    data = dataset.data.numpy()
    coords, features = data[:, :_N_COORD_COLS], data[:, _N_COORD_COLS:]
    features = zscore(features, axis=1)
    new_data = torch.tensor(np.column_stack([coords, features]), dtype=torch.float64)
    return replace(dataset, data=new_data)




def apply_pca(dataset: "UserDataset", n_components, random_state: int = 42) -> "UserDataset":
    """
    Reduce the feature columns (beyond X,Y,Z) via PCA, keeping
    coordinate columns untouched.

    Args:
        n_components: Either an int (fixed number of components), or a
            float in (0, 1) -- e.g. 0.99 -- meaning "keep as many
            components as needed to explain at least this fraction of
            total variance" (sklearn's PCA supports this directly).
        random_state: Pass `config.seed` for reproducibility.
    """
    data = dataset.data.numpy()
    coords, features = data[:, :_N_COORD_COLS], data[:, _N_COORD_COLS:]
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(features)
    new_data = torch.tensor(np.column_stack([coords, reduced]), dtype=torch.float64)
    print(f"Applied PCA: reduced features from {features.shape[1]} to {reduced.shape[1]} columns.")
    print(f"Final dataset shape: {new_data.shape}.")
    return replace(dataset, data=new_data)



def preprocess_dataset(dataset: "UserDataset", pca_variance: float = 0.99, random_state: int = 42) -> "UserDataset":
    """
    Default preprocessing entry point: check the dataset's shape, and
    apply cleanup only if there are feature columns beyond X, Y, Z.

    - Coordinate-only datasets (shape[1] <= 3, e.g. synthetic data, or
      a dataset already compressed via `compress_to_pseudo_label`):
      returned unchanged -- there is nothing to preprocess, since there
      are no feature/spectral columns.
    - Datasets with feature columns (shape[1] > 3): standardize, then
      reduce dimensionality via PCA, keeping enough components to
      explain `pca_variance` fraction of total variance. Applied
      uniformly regardless of which inversion method will be used later.

    If you additionally want a GMM-derived pseudo-label (needed for
    PRE_SEGMENTED on real data), call `compress_to_pseudo_label`
    separately afterward -- that's a distinct, explicit choice, not
    bundled into this default cleanup step.

    Args:
        dataset: A UserDataset, as returned by `select_dataset`.
        pca_variance: Fraction of variance to retain (passed as PCA's
            `n_components`), e.g. 0.99.
        random_state: Pass `config.seed` for reproducibility.
    """
    if dataset.data.shape[1] <= 3:
        print("Dataset has only spatial data (X, Y, Z) -- no preprocessing required.")
        return dataset

    print(f"Dataset has {dataset.data.shape[1]} columns (coordinate + features) -- standardizing + PCA (variance={pca_variance}).")
    dataset = run_preprocessing_steps(dataset, steps=[standardize])
    dataset = apply_pca(dataset, n_components=pca_variance, random_state=random_state)
    
    return dataset


def run_preprocessing_steps(dataset: "UserDataset", steps: List[PreprocessFn]) -> "UserDataset":
    """Apply a sequence of preprocessing steps, in order. Empty list == no-op,
    i.e. preprocessing is fully optional."""
    for step in steps:
        dataset = step(dataset)
    return dataset