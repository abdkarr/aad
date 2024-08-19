import numpy as np

from .typing import RNG_TYPE
from ._input_checks import _check_rng

def gen_gt_labels(n_classes: int, n_data_points: int, rng: RNG_TYPE = None):
    rng = _check_rng(rng)

    class_priors = rng.random(n_classes)
    class_priors /= np.sum(class_priors)
    return rng.choice(n_classes, n_data_points, p=class_priors) + 1, class_priors