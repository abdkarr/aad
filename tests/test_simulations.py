import numpy as np

import aad


def test_gen_adversaries():
    # This test include stochasticity, its parameters need to be big enough
    rng = np.random.default_rng()
    n_classes = 3
    n_tasks = 1000  # Make sure this is big enough
    n_adversaries = 200  # Make sure this is big enough

    gt_labels = rng.integers(1, n_classes + 1, n_tasks)

    p_target = 0.1
    p_camo = 0.1
    responses, targeted_tasks = aad.simulations.gen_adversaries(
        gt_labels, n_adversaries, target_frac=p_target, camo_obs=p_camo
    )

    assert isinstance(responses, np.ndarray)
    assert responses.shape == (n_adversaries, n_tasks)
    assert np.sum(targeted_tasks) == np.floor(p_target * n_tasks)

    # Assert responses are the same for all attackers
    target_responses = responses[:, targeted_tasks == 1]
    assert (target_responses == target_responses[0]).all()

    # Assert targeted responses are not equal to ground truth
    assert (target_responses[0] != gt_labels[targeted_tasks == 1]).all()

    # Assert observation probability for camouflage
    p_camo_hat = np.mean(
        np.count_nonzero(responses[:, targeted_tasks == 0])
        / (n_adversaries * np.sum(targeted_tasks == 0))
    )
    assert np.abs(p_camo_hat - p_camo) <= 0.01


def test_gen_confusion_mat():
    n_classes = 5
    reliability = 2
    confusion_mat = aad.simulations.gen_confusion_mat(n_classes, reliability)

    assert confusion_mat.shape == (n_classes, n_classes)

    # This test include stochasticity, its parameters need to be big enough
    n_annotators = 20000
    mean_confusion_mat = np.mean(
        np.array(
            [
                aad.simulations.gen_confusion_mat(n_classes, reliability)
                for i in range(n_annotators)
            ]
        ),
        axis=0,
    )

    is_realibility_correct = True
    for k1 in range(n_classes):
        for k2 in range(n_classes):
            if k1 == k2:
                continue

            is_realibility_correct &= (
                np.abs(
                    mean_confusion_mat[k1, k1] / mean_confusion_mat[k2, k1]
                    - reliability
                )
                < 0.1
            )

    assert is_realibility_correct


def test_gen_worker_labels():
    n_classes = 5
    reliability = 2
    n_tasks = 1000
    p_obs = 0.1
    n_workers = 100
    rng = np.random.default_rng()

    confusion_mats = [
        aad.simulations.gen_confusion_mat(n_classes, reliability)
        for _ in range(n_workers)
    ]
    gt_labels = rng.integers(1, n_classes + 1, n_tasks)
    labels = aad.simulations.gen_worker_labels(gt_labels, confusion_mats, p_obs)

    assert labels.shape == (n_workers, n_tasks)
    assert (np.unique(labels[labels > 0]) == np.arange(1, n_classes + 1)).all()
    assert np.abs(np.mean(labels > 0) - p_obs) < 0.01
