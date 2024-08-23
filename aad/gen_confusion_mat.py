import numpy as np

from numba import njit

from .typing import RNG_TYPE, AnnotatorType
from ._input_checks import _check_rng

@njit
def _gen_confusion_mat_nb(
        n_classes: int, n_annotators: int, annot_type: AnnotatorType, reliability: float, 
        rng = np.random.Generator
    ) -> np.array:
    confusion_mats = np.zeros((n_classes, n_classes, n_annotators))

    for a in range(n_annotators):

        # Annotator select the same class for all data points
        if annot_type == AnnotatorType.SAME:
            i = rng.integers(0, n_classes)
            confusion_mats[i, :, a] = 1
        else:
            for k in range(n_classes):
                probs = rng.random(n_classes)

                # Annotator select the correct class better than random
                if annot_type == AnnotatorType.RELIABLE:
                    j = np.argmax(probs)
                    probs[k], probs[j] = probs[j], probs[k]*reliability*1.001

                # Annotator tries to pick a class that is different than the true class
                elif annot_type == AnnotatorType.ADVERSE:
                    j = np.argmax(probs)
                    while j == k:
                        probs = rng.random(n_classes)
                        j = np.argmax(probs)

                # Annotator could be reliable or not, no need to modify probs
                elif annot_type == AnnotatorType.NOCONTROL:
                    pass

                probs /= np.sum(probs)
                confusion_mats[:, k, a] = probs

    return confusion_mats

def gen_confusion_mat(
        n_classes: int, n_annotators: int, 
        annotator_type: AnnotatorType | str = AnnotatorType.RELIABLE, 
        reliability: float | None = 1, rng: RNG_TYPE = None
    ) -> np.array:

    if isinstance(annotator_type, str):
        try:
            annotator_type = AnnotatorType(annotator_type)
        except ValueError:
            raise ValueError("Invalid annotator type.")

    if annotator_type == AnnotatorType.RELIABLE:
        if reliability is None:
            reliability = 1
        elif reliability < 1:
            raise ValueError("Reliability must be >= 1.")

    rng = _check_rng(rng)    

    # Annotator randomly picks a value
    if annotator_type == AnnotatorType.UNIFORM:
        return np.ones((n_classes, n_classes, n_annotators))/n_classes

    return _gen_confusion_mat_nb(n_classes, n_annotators, annotator_type, reliability, rng)