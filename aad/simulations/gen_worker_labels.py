import numpy as np
import numpy.typing as npt

from aad.typing import RNGType
from aad._input_checks import _check_rng


def gen_worker_labels(
    gt_labels: npt.NDArray,
    confusion_mats: list[npt.NDArray],
    p_obs: float,
    rng: RNGType = None,
):
    """Generate labels for a simulated crowdsourcing workers.

    The function generates labels based on Dawid-Skene model where each worker
    has a confusion matrix.

    ??? Example
        The following code first generates random ground truth labels for 1000
        tasks where each task belongs to one of 5 classes. Responses for 100
        workers are then generated after drawing their confusion matrices.

        ```python
        import numpy as np
        import aad

        n_classes = 5
        reliability = 2
        n_tasks = 1000
        p_obs = 0.1
        n_workers = 100
        rng = np.random.default_rng()

        # Generate random ground truth labels
        gt_labels = rng.integers(1, n_classes + 1, n_tasks)

        confusion_mats = [
            aad.simulations.gen_confusion_mat(n_classes, reliability)
            for _ in range(n_workers)
        ]
        labels = aad.simulations.gen_worker_labels(gt_labels, confusion_mats, p_obs)
        ```

    Parameters
    ----------
    gt_labels
        $(N, )$ dimensional vector where `gt_labels[i]` is the ground truth
        label of $i$th taks.
    confusion_mats
        Length $M$ list of workers' confusion matrices, where `confusion_mats[i]` is $i$th
        worker confusion matrix.
    p_obs
        Observation probability. Each worker labels only a `p_obs` fraction of
        tasks.
    rng
        Random number generator to use to draw labels.

    Returns
    -------
    response_mat
        $(M, N)$ dimensional matrix where `response_mat[i, j]` is the label provided
        by $i$th worker for $j$th task. `response_mat[i, j] = 0` indicates that
        no label is given by $i$th worker for $j$th task.
    """
    rng = _check_rng(rng)

    n_tasks = len(gt_labels)
    n_workers = len(confusion_mats)
    class_ids = np.unique(gt_labels)
    class_to_idx = {k: i for i, k in enumerate(class_ids)}

    # Generates response matrix per worker and class based on worker confusion matrices
    response_mat = np.zeros((n_workers, n_tasks))
    for k in class_ids:
        tasks_in_k = gt_labels == k
        n_tasks_in_k = np.sum(tasks_in_k)
        for w in range(n_workers):
            response_mat[w, tasks_in_k] = rng.choice(
                class_ids, size=n_tasks_in_k, p=confusion_mats[w][:, class_to_idx[k]]
            )

    # Mask response matrix based on observation probabilities
    if p_obs < 1:
        response_mat *= rng.binomial(1, p_obs, size=(n_workers, n_tasks))

    return response_mat
