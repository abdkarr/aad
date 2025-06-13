import numpy as np

import numpy.typing as npt

from scipy import stats

def _apply(response_mat: npt.NDArray) -> npt.NDArray:
    """Apply majority voting to crowdsourced dataset.

    !!! Example
        The following code apply majority voting to find labels for tasks in a 
        crowdsourcing problem. We assume we are given `response_mat`, which is 
        a $(M, N)$ dimension numpy array representing response matrix of a 
        crowdsourced dataset.  

        ```
        import aad

        mv_labels = aad.majority_voting(response_mat)
        ```

    Parameters
    ----------
    response_mat
        $(M, N)$ dimensional matrix where `response_mat[i, j]` is the label provided
        by $i$th worker for $j$th task. `response_mat[i, j] = 0` is assumed to 
        indicate no label is given by $i$th worker for $j$th task.

    Returns
    -------
    labels : npt.NDArray
        $(N, )$ dimensional array where `labels[i]` is the label of $i$th task
        estimated by majority voting.
    """
    response_mat = np.ma.masked_array(response_mat, response_mat == 0)
    labels_hat = stats.mode(response_mat, axis=0).mode
    
    return labels_hat