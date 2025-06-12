import numpy as np
import numpy.typing as npt

from scipy import sparse

from . import majority_voting


def _apply(response_mat: npt.NDArray, max_iter: int = 1000):
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

    # EM iterations
    q = np.ones((n_tasks, n_classes))/n_classes
    for i in range(max_iter):
        confusion_mats[confusion_mats == 0] = np.finfo(float).eps
        confusion_mats[np.isnan(confusion_mats)] = np.finfo(float).eps

        # M Step
        q_new = np.array(
            [onehot_labels[m] @ np.log(confusion_mats[m]) for m in range(n_workers)]
        ).sum(axis=0)
        q_new += np.log(class_priors)
        q_new = np.exp(q_new)
        q_new /= np.sum(q_new, axis=1, keepdims=True)

        # E Step
        class_priors_new = np.sum(q_new, axis=0)
        class_priors_new /= np.sum(class_priors_new)

        confusion_mats_new = np.array([onehot_labels[m].T @ q_new for m in range(n_workers)])
        confusion_mats_new /= np.sum(confusion_mats_new, axis=2, keepdims=True)

        err_q = np.linalg.norm(q_new - q)
