import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def cluster_acc(Y_pred, Y, ignore_label=None):
    """
    Rearrange predicted cluster labels to best match true labels (via the
    Hungarian algorithm), then compute accuracy under that best matching.
    Necessary for unsupervised methods (GMM, K-means, ...) whose cluster
    numbering is arbitrary and has no reason to align with ground truth.

    Returns:
        (row_ind, col_ind, accuracy, cost_matrix) -- accuracy is a
        FRACTION (0-1), a torch scalar tensor, NOT a percentage.
    """
    if ignore_label is not None:
        index = Y != ignore_label
        Y = Y[index]
        Y_pred = Y_pred[index]

    assert Y_pred.shape == Y.shape
    D = int(max(Y_pred.max(), Y.max()).item())
    w = torch.zeros((D, D))
    for i in range(Y_pred.shape[0]):
        w[int(Y_pred[i].item()) - 1, int(Y[i].item()) - 1] += 1

    row_ind, col_ind = linear_sum_assignment(w.max().item() - w.numpy())
    accuracy = w[row_ind, col_ind].sum() / Y_pred.shape[0]
    return row_ind, col_ind, accuracy, w


def _internal_validity_metrics(features: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz for `labels` against `features`."""
    result = {}
    if len(np.unique(labels)) < 2:
        print(f"  [{name}] fewer than 2 distinct clusters present -- skipping internal metrics.")
        return result

    result[f"silhouette_{name}"] = silhouette_score(features, labels)
    result[f"davies_bouldin_{name}"] = davies_bouldin_score(features, labels)
    result[f"calinski_harabasz_{name}"] = calinski_harabasz_score(features, labels)

    print(f"  [{name}] silhouette={result[f'silhouette_{name}']:.4f}, "
          f"davies_bouldin={result[f'davies_bouldin_{name}']:.4f}, "
          f"calinski_harabasz={result[f'calinski_harabasz_{name}']:.4f}")
    return result


def overall_accuracies(segmentation_map, groundtruth, model_name: str) -> dict:
    """
    Direct-match accuracy (percentage, plain float) -- valid ONLY when
    the segmentation's class numbering already matches ground truth by
    construction (e.g. the model's own GemPy-derived output). For
    GMM/K-means (arbitrary numbering), use cluster_acc instead -- see
    calculate_matrix's own dispatch below.
    """
    result = {}
    seg_t = torch.as_tensor(segmentation_map, dtype=torch.float64)
    groundtruth_t = torch.as_tensor(groundtruth, dtype=torch.float64)

    num_matches = (seg_t == groundtruth_t).sum().item()
    overall_accuracy = (num_matches / groundtruth_t.shape[0]) * 100

    print(f"Overall accuracy {model_name}: {overall_accuracy:.2f}%")
    result[f"Overall accuracy {model_name}"] = overall_accuracy
    return result


def calculate_matrix(dataset, model_output, n_clusters=None, seed=42, classical_method=None):
    """
    #  labels?  features?  segmentations list       accuracy?   validity?
    #  -------  ---------  -----------------------   ---------   ---------
    #    YES       NO       [model]                    YES          NO
    #    YES       YES      [model, gmm, kmeans]        YES          YES
    #    NO        NO       [model]                      NO          NO
    #    NO        YES      [model, gmm, kmeans]          NO          YES

    Args:
        dataset: UserDataset.
        model_output: Continuous model output, rounded into a segmentation map.
        n_clusters: Number of clusters for GMM/K-means. If not given,
            inferred from the number of distinct ground-truth labels
            when labels are available; otherwise required explicitly.
        seed: Random state for GMM/K-means reproducibility.
        classical_method: If not None, also fit and score GMM/K-means
            baselines (both together) when features are available.

    Returns:
        A dict of computed metric values. ALL accuracy values (model,
        GMM, K-means alike) are stored as percentages (plain floats,
        0-100) -- consistent units throughout.
    """
    
    matrix = {}
    data_for_accuracy = {}
    segmentation_map = torch.round(model_output).numpy()
    
    if dataset.data.shape[1] < 3:
        raise ValueError("Data has fewer than 3 columns -- spatial coordinates appear to be missing.")

    elif dataset.data.shape[1] == 3:
        print("Only Spatial data is available")
        if dataset.labels is None:
            print("Groundtruth is not available -- no metrics can be computed.")
            return matrix

        print("Groundtruth is available")
        true_labels = dataset.labels
        data_for_accuracy["model"] = segmentation_map
        matrix.update(overall_accuracies(segmentation_map, true_labels, "model"))
        return matrix , data_for_accuracy

    else:
        print("Spatial + Spectral data are available.")

        if n_clusters is None:
            if dataset.labels is not None:
                n_clusters = int(dataset.labels.max().item())
            else:
                raise ValueError(
                    "n_clusters must be given explicitly when dataset.labels is None "
                    "(e.g. pass config.clusters) -- it can't be inferred without ground truth."
                )

        features = dataset.data[:, 3:].numpy()
        data_for_accuracy["model"] = segmentation_map

        if classical_method :
            gmm_labels = GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(features) + 1
            data_for_accuracy["GMM"] = gmm_labels
            kmeans_labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(features) + 1
            data_for_accuracy["Kmeans_label"] = kmeans_labels

        for key, values in data_for_accuracy.items():
            matrix.update(_internal_validity_metrics(features, values, key))

        if dataset.labels is not None:
            true_labels = dataset.labels
            print("Groundtruth is available.")

            for key, values in data_for_accuracy.items():
                if key == "model":
                    # Model's class numbering is fixed by construction -- direct match.
                    matrix.update(overall_accuracies(values, true_labels, key))
                else:
                    # GMM/K-means numbering is arbitrary -- permutation-invariant match.
                    # Converted explicitly to a tensor first (fit_predict returns
                    # numpy), and the fraction result is scaled to a percentage
                    # and unwrapped to a plain float -- SAME units and type as
                    # "model" above, not a raw 0-1 tensor.
                    values_t = torch.as_tensor(values, dtype=torch.float64)
                    _, _, acc_frac, _ = cluster_acc(values_t, true_labels)
                    overall_accuracy = acc_frac.item() * 100
                    print(f"Overall accuracy {key}: {overall_accuracy:.2f}%")
                    matrix[f"Overall accuracy_{key}"] = overall_accuracy
        else:
            print("Groundtruth is not available.")
            
    
    return matrix , data_for_accuracy