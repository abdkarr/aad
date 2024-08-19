import pytest
import numpy as np

from aad.gen_confusion_mat import gen_confusion_mat
from aad.typing import AnnotatorType

def test_reliable_annotator():
    n_classes = 3
    n_annotators = 2

    confusion_mat = gen_confusion_mat(n_classes, n_annotators, "reliable")
    
    assert confusion_mat.shape == (n_classes, n_classes, n_annotators)

    for m in range(n_annotators):
        for k in range(n_classes):
            assert np.argmax(confusion_mat[:, k, m].squeeze()) == k

def test_adverse_annotator():
    n_classes = 3
    n_annotators = 2

    confusion_mat = gen_confusion_mat(n_classes, n_annotators, "adverse")
    
    assert confusion_mat.shape == (n_classes, n_classes, n_annotators)

    for m in range(n_annotators):
        for k in range(n_classes):
            assert np.argmax(confusion_mat[:, k, m].squeeze()) != k

def test_nocontrol_annotator():
    n_classes = 3
    n_annotators = 100 # make sure this is large enough

    confusion_mat = gen_confusion_mat(n_classes, n_annotators, "no-control")
    
    assert confusion_mat.shape == (n_classes, n_classes, n_annotators)

    n_max_is_same_as_gt = 0
    for m in range(n_annotators):
        for k in range(n_classes):
            if np.argmax(confusion_mat[:, k, m].squeeze()) == k:
                n_max_is_same_as_gt += 1

    total_cases = n_classes*n_annotators
    assert n_max_is_same_as_gt < total_cases
    assert n_max_is_same_as_gt > 0

def test_uniform_annotator():
    n_classes = 3
    n_annotators = 2

    confusion_mat = gen_confusion_mat(n_classes, n_annotators, "uniform")
    
    assert confusion_mat.shape == (n_classes, n_classes, n_annotators)
    assert np.all(np.abs(confusion_mat - 1/n_classes) < 1e-8)

def test_same_annotator():
    n_classes = 3
    n_annotators = 2

    confusion_mat = gen_confusion_mat(n_classes, n_annotators, "same")
    
    assert confusion_mat.shape == (n_classes, n_classes, n_annotators)
    
    for m in range(n_annotators):
        for k in range(n_classes):
            indx = np.where(confusion_mat[:, k, m].squeeze() != 0)[0]
            assert len(indx) == 1
            assert confusion_mat[indx, k, m] == 1

def test_reproducibility():
    n_classes = 3
    n_annotators = 2

    confusion_mat1 = gen_confusion_mat(n_classes, n_annotators, "reliable", rng=1)
    confusion_mat2 = gen_confusion_mat(n_classes, n_annotators, "reliable", rng=1)

    assert np.all(np.abs(confusion_mat1 - confusion_mat2) < 1e-8)

def test_annotator_type_arg():
    n_classes = 3
    n_annotators = 2
    wrong_annot_type = "wrong"

    with pytest.raises(ValueError):
        confusion_mat = gen_confusion_mat(n_classes, n_annotators, wrong_annot_type)


def test_reliablity_arg():
    n_classes = 3
    n_annotators = 2
    valid_reliability = 2 # >=1 are valid

    gen_confusion_mat(n_classes, n_annotators, "reliable", reliability=valid_reliability)

    invalid_reliability = 0.9
    with pytest.raises(ValueError):
        gen_confusion_mat(n_classes, n_annotators, "reliable", reliability=invalid_reliability)

