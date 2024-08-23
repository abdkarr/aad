import numpy as np

import aad 

def test_random_one_group_label_generation():
    # This test include stochasticity, its parameters need to be big enough
    rng = np.random.default_rng()
    n_classes = 3
    n_data_points = 1000 # Make sure this is big enough
    n_adversaries = 200 # Make sure this is big enough
    
    gt_labels, _ = aad.gen_gt_labels(n_classes, n_data_points, rng)

    for same_resp_per_class in [True, False]:
        for p_attacked in [0.2, 1]:
            responses = aad.gen_adversary_labels(
                gt_labels, n_adversaries, "random", p_attacked=p_attacked, p_obs=1, 
                same_resp_per_class=same_resp_per_class, rng=rng
            )

            assert isinstance(responses, np.ndarray)
            assert responses.shape == (n_adversaries, n_data_points)

            # Find attacked points and assert their fraction
            attacked_points = responses[0, :] == responses[1, :]
            for a in range(2, n_adversaries):
                attacked_points = attacked_points & (responses[0, :] == responses[a, :])
            assert np.abs(np.mean(attacked_points) - p_attacked) < 1e-2
        
            # Assert responses are the same for all attackers
            responses = responses[:, attacked_points]
            assert np.isclose(responses, responses[0]).all()

            # Assert same response per class argument works correctly
            if same_resp_per_class:
                for k in range(1, n_classes+1):
                    class_response = responses[0, gt_labels[attacked_points] == k]
                    assert (class_response == class_response[0]).all()
                    assert (class_response != k).all()

    # Assert observation probability argument work correctly
    p_obs = 0.2
    responses = aad.gen_adversary_labels(
        gt_labels, n_adversaries, "random", p_attacked=p_attacked, p_obs=0.2, 
        same_resp_per_class=same_resp_per_class, rng=rng
    )
    p_obs_hat = np.mean(np.count_nonzero(responses, axis=1)/n_data_points)
    assert np.abs(p_obs_hat - p_obs) <= 0.01