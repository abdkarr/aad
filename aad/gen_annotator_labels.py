import numpy as np

from numba import njit

from .typing import RNG_TYPE
from ._input_checks import _check_rng

# @njit
def _gen_annotator_labels_nb(
        confusion_mats: np.array, gt_labels: np.array, p_obs: np.array, 
        ensure_all_classes: bool, rng: np.random.Generator
    ) -> np.ndarray:

    n_annotators = confusion_mats.shape[2]
    n_classes = confusion_mats.shape[0]
    n_data_points = gt_labels.shape[0]

    # Draw the initial labels for annotators
    labels = np.zeros((n_annotators, n_data_points))
    for k in range(0, n_classes):
        idx = np.where(gt_labels == k+1)[0]
        for a in range(n_annotators):
            n_samples = len(idx)
            outcomes = np.arange(1, n_classes+1)

            # Since rng.choice doesn't work with p argument in numba
            labels[a, idx] = outcomes[
                np.searchsorted(np.cumsum(confusion_mats[:, k, a]), rng.random(n_samples))
            ]

    for a in range(n_annotators):
        # Set some labels to 0 indicating that annotator did not label the data point
        curr_labels = labels[a, :]*rng.binomial(1, p_obs[a], n_data_points)

        # Ensure the annotator provides labels for all classes
        if ensure_all_classes:
            idx = np.where(curr_labels > 0)[0]
            nonzero_labels = curr_labels[idx]
            existing_classes = np.unique(nonzero_labels)
            n_labels = idx.shape[0]
            
            if len(existing_classes) < n_classes:
                for k in range(1, n_classes + 1):
                    if k in existing_classes:
                        continue

                    idx = np.where(labels[a, :] == k)[0]
                    rng.shuffle(idx)
                    n_points_to_add = int(np.ceil(0.1*n_labels))
                    if n_data_points < len(idx):
                        idx = idx[:n_points_to_add]
                    curr_labels[idx] = k 

        labels[a, :] = curr_labels

    return labels

def gen_annotator_labels(
        confusion_mats: np.ndarray, gt_labels: np.ndarray, p_obs: float | np.ndarray = 1, 
        ensure_all_classes: bool = True, rng: RNG_TYPE = None
    ) -> np.ndarray:

    rng = _check_rng(rng)

    n_annotators = confusion_mats.shape[2]

    if np.ndim(p_obs) == 0:
        p_obs = p_obs*np.ones(n_annotators)

    return _gen_annotator_labels_nb(
        confusion_mats, gt_labels, p_obs, ensure_all_classes, rng
    )