import numpy.typing as npt

import numpy as np
import numpy.typing as npt

from numba import njit
from scipy import sparse

from .calc_agreement_mat import calc_agreement_mat


# @njit
def _calc_edge_weights(
    response_mat: npt.NDArray,
    workers: npt.NDArray,
    tasks: npt.NDArray,
    responses: npt.NDArray,
) -> npt.NDArray:
    """
    Calculate edge weights of the bipartite graph representation of a crowdsourced
    dataset based on agreement rates and co-labeling.
    """

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


def _construct_biadj_mat(
    response_mat: npt.NDArray, kind: str = "binary"
) -> npt.NDArray:
    """
    Construct the bi-adjacency matrix of the bipartite graph representation
    of a crowdsourced dataset.
    """

    n_workers, n_tasks = response_mat.shape

    workers, tasks = np.nonzero(response_mat)
    responses = response_mat[workers, tasks]

    if kind == "binary":
        weights = 1
    elif kind == "weighted":
        weights = _calc_edge_weights(response_mat, workers, tasks, responses)

    biadj_mat = np.zeros((n_workers, n_tasks))
    biadj_mat[workers, tasks] = weights

    return biadj_mat


class MinTree:
    """
    A tree structure that can be used to efficiently find the minimum value in
    an array while allowing updates to values of array entries. . Implementation
    is based on Fraudar [1].

    References
    ----------
    [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
    camouflage." Proceedings of the 22nd ACM SIGKDD international conference on
    knowledge discovery and data mining. 2016.
    """

    def __init__(self, values: npt.ArrayLike):
        """Initializer.

        Parameters
        ----------
        values
            Array values from which the tree is constructed
        """
        self.tree_height = int(np.ceil(np.log2(len(values))))
        self.n_leaves = 2**self.tree_height
        self.n_branches = self.n_leaves - 1
        self.n_nodes = self.n_branches + self.n_leaves

        self.nodes = [np.inf] * self.n_nodes
        # Leaf nodes
        for i in range(len(values)):
            self.nodes[self.n_branches + i] = values[i]

        # Parents carry the smallest value of their children
        for i in reversed(range(self.n_branches)):
            self.nodes[i] = min(self.nodes[2 * i + 1], self.nodes[2 * i + 2])

    def get_min_value(self):
        """
        Finds the leaf with the smallest value
        """
        curr = 0
        for i in range(self.tree_height):
            if self.nodes[2 * curr + 1] <= self.nodes[2 * curr + 2]:
                curr = 2 * curr + 1
            else:
                curr = 2 * curr + 2

        # return leaf index and value
        return (curr - self.n_branches, self.nodes[curr])

    def update_value(self, leaf, delta):
        """
        Update a value of leaf and its parent nodes in the tree
        """
        curr = self.n_branches + leaf
        self.nodes[curr] += delta
        for i in range(self.tree_height):
            parent = (curr - 1) // 2
            new_parent_val = min(self.nodes[2 * parent + 1], self.nodes[2 * parent + 2])

            if self.nodes[parent] == new_parent_val:
                break

            self.nodes[parent] = new_parent_val
            curr = parent


def _peel(biadj_mat: npt.NDArray):
    """
    Run peeling algorithm on a bipartite graph. Implementation is based on
    Fraudar [1].

    References
    ----------
    [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
    camouflage." Proceedings of the 22nd ACM SIGKDD international conference on
    knowledge discovery and data mining. 2016.
    """
    n_workers, n_tasks = biadj_mat.shape

    worker_degrees = np.sum(biadj_mat, axis=1)
    task_degrees = np.sum(biadj_mat, axis=0)

    # Construct Minimum search trees for peeling algorithm
    workers_tree = MinTree(worker_degrees)
    tasks_tree = MinTree(task_degrees)

    # Peeling algorithm - inputs
    biadj_lil = sparse.lil_array(biadj_mat)
    biadj_lil_t = sparse.lil_array(biadj_mat.T)
    workers = set(range(0, n_workers))
    tasks = set(range(0, n_tasks))

    # Peeling algorithm - outputs
    workers_order = []
    tasks_order = []

    # Peeling algorithm - iterations
    iter = 0
    while workers and tasks:
        iter += 1

        # Find the node with the minimum degree in the bipartite graph
        min_worker, min_worker_degree = workers_tree.get_min_value()
        min_task, min_task_degree = tasks_tree.get_min_value()

        if min_worker_degree <= min_task_degree:  # Peel the worker
            for task in biadj_lil.rows[min_worker]:
                change = biadj_mat[min_worker, task]
                tasks_tree.update_value(task, -change)

            workers.remove(min_worker)
            workers_tree.update_value(min_worker, np.inf)
            workers_order.append(min_worker)

        else:  # Peel the task
            for worker in biadj_lil_t.rows[min_task]:
                change = biadj_mat[worker, min_task]
                workers_tree.update_value(worker, -change)

            tasks.remove(min_task)
            tasks_tree.update_value(min_task, np.inf)
            tasks_order.append(min_task)

    # Add the last remaining node to order arrays
    if len(workers) == 0:
        tasks_order.append(list(tasks)[0])
    if len(tasks) == 0:
        workers_order.append(list(workers)[0])

    return workers_order, tasks_order


def _calc_adversary_scores(workers_order, tasks_order):
    """
    Calculate adversary scores of workers and tasks based on their peeling order
    """
    n_workers = len(workers_order)
    n_tasks = len(tasks_order)

    worker_scores = np.zeros(n_workers)
    task_scores = np.zeros(n_tasks)

    for i, w in enumerate(workers_order):
        worker_scores[w] = i + 1
    for i, t in enumerate(tasks_order):
        task_scores[t] = i + 1

    worker_scores = worker_scores / n_workers
    task_scores = task_scores / n_tasks

    return worker_scores, task_scores


def detect_attacks(
    response_mat: npt.NDArray, kind: str = "binary"
) -> tuple[npt.NDArray, npt.NDArray]:
    """Detect adversarial workers and their targeted tasks.

    The detection is performed by first constructing a bipartite graph $G=(W, T, E)$
    from the response matrix of a crowdsourced dataset. In $G$, $W$ and $T$ are
    the nodes representing workers and tasks and edges connect a worker and
    a task if that worker provided label for that task. An edge weight mechanism
    that employs workers agreement rates and co-labeling is also provided.
    The constructed bipartite graph is then processed using a peeling algorithm
    implemented based on [1]. The peeling order of workers and tasks are then
    used to calculate adversary scores such that the higher scores indicate
    higher likelihood for a worker to be adversarial or a task to be targeted.
    Further details can be found in [2].

    Parameters
    ----------
    response_mat
        $(M, N)$ dimensional matrix where `response_mat[i, j]` is the label provided
        by $i$th worker for $j$th task. `response_mat[i, j] = 0` is assumed to 
        indicate no label is given by $i$th worker for $j$th task.
    kind
        Kind of bipartite graph to constructed. Must be either "binary" or "weighted".
        In latter case, the edges of the bipartite graph are weighted as described 
        in [2].

    Returns
    -------
    worker_scores 
        $(M, )$ dimensional array where `worker_scores[i]` is the adversary 
        score of $i$th worker indicating the likelihood of $i$ being an adversary.
    task_scores
        $(N, )$ dimensional array where `task_scores[i]` is the adversary score
        of $i$th task indicating likelihood of $i$ being a targeted task.

    References
    ----------
    [1] Hooi, Bryan, et al. "Fraudar: Bounding graph fraud in the face of
    camouflage." Proceedings of the 22nd ACM SIGKDD international conference on
    knowledge discovery and data mining. 2016.

    [2] Karaaslanli, Abdullah, Panagiotis A. Traganitis, and Aritra Konar.
    "Identifying Adversarial Attacks in Crowdsourcing via Dense Subgraph Detection."
    ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal
    Processing (ICASSP). IEEE, 2025.
    """

    biadj_mat = _construct_biadj_mat(response_mat, kind)
    workers_order, tasks_order = _peel(biadj_mat)
    worker_scores, task_scores = _calc_adversary_scores(workers_order, tasks_order)

    return worker_scores, task_scores
