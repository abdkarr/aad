import numpy as np

import aad

def test_return_mat_size():
    n_classes = 3
    n_annotators = 50
    n_data_points = 1000

    confusion_mats = aad.gen_confusion_mat(n_classes, n_annotators)
    gt_labels, _ = aad.gen_gt_labels(n_classes, n_data_points)

    # check returned matrix size is correct
    p_obs = 0.1
    labels = aad.gen_annotator_labels(confusion_mats, gt_labels, p_obs, 
                                      ensure_all_classes=True)
    assert labels.shape == (n_annotators, n_data_points)

    labels = aad.gen_annotator_labels(confusion_mats, gt_labels, p_obs, 
                                      ensure_all_classes=False)
    assert labels.shape == (n_annotators, n_data_points)

def test_labels():
    n_classes = 3
    n_annotators = 50
    n_data_points = 1000

    confusion_mats = aad.gen_confusion_mat(n_classes, n_annotators)
    gt_labels, _ = aad.gen_gt_labels(n_classes, n_data_points)

    # check labels are in [1, n_classes] when annotators respond to all data points
    p_obs = 1.0
    labels = aad.gen_annotator_labels(confusion_mats, gt_labels, p_obs, 
                                      ensure_all_classes=False)
    assert labels.shape == (n_annotators, n_data_points)
    assert np.all(np.unique(labels) == np.arange(1, n_classes + 1))

    # check labels are in [0, n_classes] when annotators respond to some data points
    p_obs = 0.5
    labels = aad.gen_annotator_labels(confusion_mats, gt_labels, p_obs, 
                                      ensure_all_classes=False)
    assert labels.shape == (n_annotators, n_data_points)
    assert np.all(np.unique(labels) == np.arange(n_classes + 1))


def test_observation_prob():
    n_classes = 3
    n_annotators = 10000 # Ensure this is large enough for statistical power
    n_data_points = 1000

    confusion_mats = aad.gen_confusion_mat(n_classes, n_annotators)
    gt_labels, _ = aad.gen_gt_labels(n_classes, n_data_points)

    for p_obs in [0.2, 0.4, 0.6, 0.8, 1]:
        labels = aad.gen_annotator_labels(confusion_mats, gt_labels, p_obs, 
                                          ensure_all_classes=False, rng=1)
        
        assert labels.shape == (n_annotators, n_data_points)

        p_obs_hat = np.mean(np.count_nonzero(labels, axis=1)/n_data_points)

        assert np.abs(p_obs_hat - p_obs) <= 0.01

def test_ensure_all_classes():
    n_classes = 10
    n_annotators = 50
    n_data_points = 1000

    confusion_mats = aad.gen_confusion_mat(n_classes, n_annotators)
    gt_labels, _ = aad.gen_gt_labels(n_classes, n_data_points)

    labels = aad.gen_annotator_labels(confusion_mats, gt_labels, 0.05, 
                                        ensure_all_classes=True)
    
    for a in range(n_annotators):
        curr_labels = np.unique(labels[a, :])
        for k in range(n_classes):
            assert k in curr_labels
        