import numpy as np

import numpy.typing as npt


def _sigmoid(x, shift=0.2, scale=5):
    return np.exp(-(x - shift) * scale * 10) / (1 + np.exp(-(x - shift) * scale * 10))

def _apply(response_mat, worker_scores, task_scores, apply_sigmoid = True, scale=5, shift=0.3):
    n_tasks = response_mat.shape[1]

    class_ids = np.unique(response_mat)
    if 0 in class_ids:
        class_ids = class_ids[class_ids != 0]

    mv_labels = np.zeros(n_tasks)
    for t in range(n_tasks):
        t_workers = np.where(response_mat[:, t])[0]
        
        if apply_sigmoid:
            weights = np.array(
                [
                    1
                    - _sigmoid(1 - worker_scores[w], scale=scale, shift=shift)
                    * _sigmoid(1 - task_scores[t], scale=scale, shift=shift)
                    for w in t_workers
                ]
            )
        else:
            weights = np.array([worker_scores[w]*task_scores[t] for w in t_workers])

        t_labels = response_mat[t_workers, t]

        max_weight = -1
        # rng.shuffle(class_ids)
        for c in class_ids:
            c_weight = np.sum(weights[t_labels == c])
            if c_weight > max_weight:
                max_weight = c_weight
                best_c = c

        mv_labels[t] = best_c

    return mv_labels