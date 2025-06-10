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
        gt_labels, n_adversaries, p_target, p_camo, rng
    )

    assert isinstance(responses, np.ndarray)
    assert responses.shape == (n_adversaries, n_tasks)
    assert np.sum(targeted_tasks) == np.floor(p_target * n_tasks)

    # Assert responses are the same for all attackers
    target_responses = responses[:, targeted_tasks]
    assert (target_responses == target_responses[0]).all()

    # Assert targeted responses are not equal to ground truth
    assert (target_responses[0] != gt_labels[targeted_tasks]).all()

    # Assert observation probability for camouflage
    p_camo_hat = np.mean(
        np.count_nonzero(responses[:, ~targeted_tasks])
        / (n_adversaries * np.sum(~targeted_tasks))
    )
    assert np.abs(p_camo_hat - p_camo) <= 0.01
