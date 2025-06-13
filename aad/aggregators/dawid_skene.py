import numpy as np
import numpy.typing as npt

from scipy import sparse

from . import majority_voting


def _apply(response_mat: npt.NDArray, max_iter: int = 1000, tol: float = 1e-6) -> dict:
    r"""Apply Dawid-Skene aggregation to a crowdsourced dataset.

    This function use majority-voting for initializing the Expectation-Maximization (EM)
    algorithm.

    !!! Example
        The following code applies Dawid-Skene to estimate task labels of RTE
        dataset.

        ```
        from pathlib import Path

        import aad

        data_dir = Path(".")
        response_mat, gt_labels = aad.datasets.read_rte(data_dir)
        ds_out = aad.dawid_skene(response_mat)
        ```

    Parameters
    ----------
    response_mat
        $(M, N)$ dimensional matrix where `response_mat[i, j]` is the label provided
        by $i$th worker for $j$th task. `response_mat[i, j] = 0` is assumed to
        indicate no label is given by $i$th worker for $j$th task.
    max_iter
        Maximum number EM iterations
    tol
        Tolarance to use convergence. At each EM iteration, the changes in model
        parameters and task classification probabilities are calculated. If the
        change is smaller than `tol`, EM is deemed as converged.

    Returns
    -------
    out : dict
        Output dictionary consisting of following elements:

        - *"labels"*: $(N, )$ dimensional array where `out["labels"][i]` is
        the label estimated for $i$th task.
        - *"probs"*: $(N, K)$ dimensional array where `out["probs"][i, j]`
        is the probability of $i$th task being $j$th class.
        - *"confusion_mats"*: $(M, K, K)$ dimensional array where
        `out["confusion_mats][i, :, :]` is the estimated confusion matrix of 
        $i$th worker.
        - *"class_priors"*: $(K, )$ dimensional array of estimated class priors.
    """

    n_workers, n_tasks = response_mat.shape

    responses = response_mat[response_mat > 0]
    class_ids = np.unique(responses)
    n_classes = len(class_ids)
    class_to_idx = {c: i for i, c in enumerate(class_ids)}

    # Initialize structure for EM algorithm
    onehot_labels = []
    confusion_mats = []
    mv_labels = majority_voting._apply(response_mat)
    for m in range(n_workers):
        m_tasks = np.where(response_mat[m, :])[0]
        m_responses = np.array([class_to_idx[l] for l in response_mat[m, m_tasks]])

        onehot_labels.append(
            sparse.csr_array(
                ([1] * len(m_tasks), (m_tasks, m_responses)), shape=(n_tasks, n_classes)
            )
        )

        # Calculate m's confusion matrix from MV labels
        m_confusion = np.zeros((n_classes, n_classes))
        m_tasks_gt = mv_labels[m_tasks]
        for k1 in range(n_classes):
            m_tasks_in_k1 = m_tasks_gt == class_ids[k1]
            for k2 in range(n_classes):
                # Number of tasks in class k1 that is labeled as k2 by mth worker
                m_confusion[k2, k1] = np.sum(m_responses[m_tasks_in_k1] == k2)

            if np.sum(m_confusion[:, k1]) == 0:
                m_confusion[:, k1] = np.ones(n_classes)
            m_confusion[:, k1] /= np.sum(m_confusion[:, k1])

        confusion_mats.append(m_confusion)
    confusion_mats = np.array(confusion_mats)

    # Calculate class priors from MV labels
    class_priors = np.zeros(n_classes)
    for k1 in range(n_classes):
        class_priors[k1] = np.sum(mv_labels == class_ids[k1]) / n_tasks

    probs = np.ones((n_tasks, n_classes)) / n_classes

    # EM iterations
    for i in range(max_iter):
        probs_prev = probs
        confusion_mats_prev = confusion_mats
        class_priors_prev = class_priors

        # E-step
        probs = np.array(
            [
                onehot_labels[m] @ np.log(confusion_mats[m] + 1e-6)
                for m in range(n_workers)
            ]
        ).sum(axis=0)
        probs += np.log(class_priors)
        probs = np.exp(probs)
        probs /= np.sum(probs, axis=1, keepdims=True)

        # M-step
        class_priors = np.sum(class_priors, axis=0)
        class_priors /= np.sum(class_priors)

        confusion_mats = np.array(
            [onehot_labels[m].T @ probs for m in range(n_workers)]
        )
        normalizer = np.sum(confusion_mats, axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1
        confusion_mats /= normalizer

        # Check convergence
        probs_change = np.linalg.norm((probs_prev - probs).flatten()) / n_tasks
        confusion_mats_chage = (
            np.linalg.norm((confusion_mats_prev - confusion_mats).flatten()) / n_workers
        )
        class_priors_change = np.linalg.norm(
            (class_priors_prev - class_priors).flatten()
        )

        if (
            (probs_change < tol)
            & (confusion_mats_chage < tol)
            & (class_priors_change < tol)
        ):
            break

    return {
        "labels": class_ids[np.argmax(probs, axis=1)],
        "probs": probs,
        "confusion_mats": confusion_mats,
        "class_priors": class_priors,
    }
