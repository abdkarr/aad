import numpy as np
import numpy.typing as npt

from aad import typing
from aad._input_checks import _check_rng


def _draw_targeted_responses(
    targeted_tasks_gt: npt.NDArray,
    class_ids: npt.NDArray,
    n_adversaries: int,
    rng: np.random.Generator,
):
    n_targeted = len(targeted_tasks_gt)

    targeted_response = np.zeros((n_adversaries, n_targeted))
    for k in class_ids:
        # Targeted tasks in class k
        targeted_tasks_in_k = targeted_tasks_gt == k

        # Possible labels adversaries can provide for tasks in k
        possible_labels = np.setdiff1d(class_ids, k)

        # All targeted tasks in k labeled with the same label by all adversaries
        targeted_response[:, targeted_tasks_in_k] = rng.choice(possible_labels)

    return targeted_response


def _draw_camouflage_responses(
    class_ids: npt.NDArray,
    n_adversaries: int,
    n_clean: int,
    p_camo: float,
    rng: np.random.Generator,
):
    # Adversaries employs random response for clean tasks to camouflage
    clean_response = rng.choice(class_ids, size=(n_adversaries, n_clean))

    # Only a subset of clean tasks are labeled by adversaries
    clean_response *= rng.binomial(1, p_camo, size=(n_adversaries, n_clean))

    return clean_response


def gen_adversaries(
    gt_labels: npt.NDArray,
    n_adversaries: int,
    p_target: float,
    p_camo: float,
    rng: typing.RNGType = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""Generate a simulated response matrix for adversarial workers.

    The function selects a set of tasks that are considered to be targeted by 
    adversaries. Adversaries provide a wrong label to targeted points. Each 
    adversary also provide labels for `p_camo` fraction of the remaining points 
    as a way to camouflage themselves. Labeling behaviour of adversaries are 
    follows:

    For targeted tasks:

    - For a targeted task $n$ that is in class $k$, all adversaries provide the 
    same  wrong label for $n$, i.e., $\widehat{y}_{mn} = k' \neq k$ for all 
    adversaries $m$. $k'$ is the same for all targeted $n$'s in class k. This 
    can considered as adversaries trying to flip label of $n$ from $k$ to $k'$.  

    For camouflage:

    - For a non-targeted task $n$, each adversary independently provide a random 
    label with probability `p_camo`, i.e., $\widehat{y}_{mn} = k \in {\cal C}$ 
    with probability `p_camo` and $\widehat{y}_{mn} = 0$ where $\cal C$ is the 
    set of class ids. This can be considered as the least amount of camouflage 
    but the cheapest way too. 

    ??? Example
        The following code first generates random ground truth labels for 1000
        tasks where each task belongs to one of 3 classes. Responses for 20 
        adversaries are then generated. 

        ```
        import numpy as np
        import aad

        # Generate random class labels 
        rng = np.random.default_rng()
        gt_labels = rng.integers(1, 4, 1000) 

        aad.simulations.gen_adversaries(gt_labels, 20, p_target=0.1, p_camo=0.1)
        ```

    Parameters
    ----------
    gt_labels
        Ground truth labels of tasks
    n_adversaries
        Number of adversaries to produce responses for
    p_target
        Fraction of tasks that is selected as targeted tasks
    p_camo
        Probability of an attacker providing a label for a non-targeted task for 
        camouflage.
    rng
        Random number generator

    Returns
    -------
    response_mat : npt.NDArray
        $(M, N)$ dimensional generated response matrix where $M$ is the number
        of adversaries and $N$ is the number of tasks. 
    targeted_tasks : npt.NDArray
        $(N, )$ dimensional binary array where `targeted_tasks[i] = 1` if $i$th 
        task is targeted, 0 otherwise. 
    """

    # Input Checks
    rng = _check_rng(rng)

    n_tasks = len(gt_labels)
    n_targeted = int(np.floor(n_tasks * p_target))
    n_clean = n_tasks - n_targeted
    class_ids = np.unique(gt_labels)

    # Draw the set of tasks targeted by adversaries
    targeted_tasks = np.zeros(n_tasks, dtype=np.bool)
    targeted_tasks[rng.choice(n_tasks, n_targeted, replace=False)] = 1

    # Determine labels for targeted points
    targeted_tasks_gt = gt_labels[targeted_tasks]
    targeted_response = _draw_targeted_responses(
        targeted_tasks_gt, class_ids, n_adversaries, rng
    )

    # Determine labels for camouflage
    camouflage_response = _draw_camouflage_responses(
        class_ids, n_adversaries, n_clean, p_camo, rng
    )

    response_mat = np.zeros((n_adversaries, n_tasks), dtype=np.int64)
    response_mat[:, targeted_tasks] = targeted_response
    response_mat[:, ~targeted_tasks] = camouflage_response

    return response_mat, targeted_tasks.astype(np.int64)
