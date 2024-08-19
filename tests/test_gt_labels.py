import numpy as np

from aad.gen_gt_labels import gen_gt_labels

def test_gt_label_generation():
    n_classes = 3
    n_data_points = 1000

    y_gt1, gt_priors1 = gen_gt_labels(n_classes, n_data_points, rng=1)
    y_gt2, gt_priors2 = gen_gt_labels(n_classes, n_data_points, rng=1)

    # Reproducibility
    assert np.all(y_gt1 == y_gt2)
    assert np.all(gt_priors1 == gt_priors2)

    # Check shapes
    assert y_gt1.shape == (n_data_points, )
    assert gt_priors1.shape == (n_classes, )
    
    # Correct class labels
    assert np.all(np.unique(y_gt1) == np.arange(1, n_classes+1))