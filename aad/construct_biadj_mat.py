import numpy as np
import numpy.typing as npt

from numba import njit
from scipy import sparse

from .calc_agreement_mat import calc_agreement_mat


# @njit
def _calc_type1_weights(
    response_mat: npt.NDArray,
    workers: npt.NDArray,
    tasks: npt.NDArray,
    responses: npt.NDArray,
) -> npt.NDArray:
    
    agreement_mat, _ = calc_agreement_mat(response_mat)
    
    n_responses = len(responses)
    weights = np.zeros(n_responses)
    for i in range(n_responses):
        worker = workers[i]
        task = tasks[i]
        label = responses[i]

        # Find workers who labeled the current task the same as current worker
        task_labels = response_mat[:, task]
        matching_workers = np.where(task_labels == label)[0]
        matching_workers = np.setdiff1d(matching_workers, worker, assume_unique=True)

        # Calculate edge weight
        if len(matching_workers) > 0:
            weights[i] = np.mean(agreement_mat[worker, matching_workers])

    return weights

def construct_biadj_mat(
    response_mat: npt.NDArray, weight: str = "binary"
) -> npt.NDArray:
    n_workers, n_tasks = response_mat.shape

    workers, tasks = np.nonzero(response_mat)
    responses = response_mat[workers, tasks]

    if weight == "binary":
        weights = 1
    elif weight == "type1":
        weights = _calc_type1_weights(response_mat, workers, tasks, responses)

    biadj_mat = np.zeros((n_workers, n_tasks))
    biadj_mat[workers, tasks] = 1

    return biadj_mat
