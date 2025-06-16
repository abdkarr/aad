import numpy as np
import numpy.typing as npt

from aad._input_checks import _check_rng
from aad.typing import RNGType


def gen_confusion_mat(
    n_classes: int, reliability: float, rng: RNGType = None
) -> npt.NDArray:
    r"""Generate a confusion matrix for a simulated crowdsourcing worker.

    The function generates a $K \times K$ dimensional matric ${\bf \Gamma}$
    where $K$ is the number of classes and $\Gamma_{ij}$ is the probability
    of worker giving label $i$ for a task in class $j$. $k$th column of
    $\bf \Gammsa is drawn from a Dirichlet distribution with parameter
    ${\bf \alpha} = {\bf 1} + (r - 1){\bf e}_k$ where ${\bf e}_k$ is the $k$th
    standard basis of $K$-dimensional space and $r$ is reliability. As $r$ grows,
    the worker becomes more reliable in limit sense. That is
    $\mathbb{E}[\Gamma_{ii}]/\mathbb{E}[\Gamma_{ij}] = r,\ \forall j \neq i$.

    ??? Example
        The following code generates a confusion matrix for a simulated crowdsourcing 
        problem with 5 classes:

        ```python
        import aad

        n_classes = 5
        reliability = 2
        confusion_mat = aad.simulations.gen_confusion_mat(n_classes, reliability)
        ```

    Parameters
    ----------
    n_classes
        Number of classes to simulate.
    reliability
        Realibility of worker.
    rng, optional
        Random number of generator to use to draw the columns of confusion matrix.

    Returns
    -------
    confusion_mat : npt.NDArray
        Generated confusion matrix.

    Raises
    ------
    ValueError
        If `relaibility` is not positive. 
    """

    rng = _check_rng(rng)

    # Check if reliability is valid
    if reliability <= 0:
        ValueError("Parameter `realiability` must be larger than 0.")

    confusion_mat = np.zeros((n_classes, n_classes))
    for k in range(n_classes):
        alphas = np.ones(n_classes)
        alphas[k] *= reliability
        confusion_mat[:, k] = rng.dirichlet(alphas)

    return confusion_mat
