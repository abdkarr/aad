import numpy as np

from .typing import AdversaryType, RNG_TYPE
from ._input_checks import _check_rng

def _gen_random_adversaries(
        gt_labels, groups, p_obs, p_attacked, collude_groups, same_resp_per_class, 
        same_resp_across_groups, rng: np.random.Generator
    ):
    n_groups = len(groups)
    n_data_points = len(gt_labels)
    n_classes = len(np.unique(gt_labels))

    # All groups attack the same data points
    n_points_not_attacked = int(np.floor((1-p_attacked)*n_data_points))
    if collude_groups:
        points_not_attacked = rng.choice(n_data_points, n_points_not_attacked, replace=False)
    else:
        same_resp_across_groups = False 

    # All groups label the attacked data points the same way
    if same_resp_across_groups:
        common_response = np.zeros(n_data_points)
        for k in range(1, n_classes + 1):
            other_classes = np.setdiff(np.arange(1, n_classes+1), k)
            points_in_k = np.where(gt_labels == k)[0]

            # Sample responses for points in class k
            n_responses = len(points_in_k)
            if same_resp_per_class:
                # Data points from the same class are labelled the same
                n_responses = 1

            common_response[points_in_k] = rng.choice(other_classes, n_responses)
                
    responses = []
    for g in range(n_groups):
        group = groups[g]
        group_size = len(group)
        
        # Generate initial responses
        if same_resp_across_groups:
            group_responses = np.tile(common_response, (group_size, 1))
        else:
            group_responses = np.zeros((group_size, n_data_points))
            for k in range(1, n_classes+1):
                
                other_classes = np.setdiff1d(np.arange(1, n_classes+1), k)
                points_in_k = np.where(gt_labels == k)[0]
                
                if same_resp_per_class:
                    # Data points from the same class are labelled the same
                    group_responses[:, points_in_k] = rng.choice(other_classes)
                else:
                    tmp = rng.choice(other_classes, len(points_in_k))
                    group_responses[:, points_in_k] = np.tile(tmp, (group_size, 1))

        # Generate responses for data points not attacked 
        if not collude_groups:
            points_not_attacked = rng.choice(n_data_points, n_points_not_attacked, replace=False)
        group_responses[:, points_not_attacked] = rng.integers(
            1, n_classes+1, size=(group_size, n_points_not_attacked)
        )
        
        # Mask annotators responses based on observation probabilities
        for a in range(group_size):
            group_responses[a] *= rng.binomial(1, p_obs[a], n_data_points)

        responses.append(group_responses)

    return responses

def gen_adversary_labels(
        gt_labels, 
        n_annotators, 
        adversary_type: AdversaryType | str = "random",
        p_attacked: float = 0.1,
        p_obs: float = 1, 
        n_groups: int = 1,
        collude_groups: bool = False,
        same_resp_per_class: bool = False,
        same_resp_across_groups: bool = False,
        rng: RNG_TYPE = None
    ):
    """_summary_

    All attackers within group labels all data points the same way, however 
    they do not necessarily see the same data points.

    Parameters
    ----------
    n_classes : _type_
        _description_
    n_annotators : _type_
        _description_
    adversary_type : AdversaryType | str, optional
        Type of adversarial attack. It should be one of the followings:
        
        - "random": Adversary provide a random response except for a set of data
        points where they provide the same wrong answer. The set of data points 
        they attack determined by argument `p_attacked`. 
        
        By default "random".
    p_attacked : float, optional
        Fraction of data points that will be attacked by an adversary, by default 0.5.
    n_groups: int, optional
        Number of adversarial groups, by default 1.
    collude_groups: bool, optional
        Flag indicating whether adversary groups attack the same data points, by
        default `False`.
    same_resp_per_class: bool, optional
        Flag indicating whether the same wrong response given for each class of data
        by annotators within each group, by default `False`.
    same_resp_across_groups: bool, optional
        Flag indicating whether the same wrong response given by all groups. 
        Ignored if `collude_groups` is `False`. By defualt `False`.

    Raises
    ------
    ValueError
        _description_
    """
    
    # Input Checks 
    rng = _check_rng(rng)

    if isinstance(adversary_type, str):
        try:
            adversary_type = AdversaryType(adversary_type)
        except ValueError:
            raise ValueError("Invalid annotator type.")

    if np.ndim(p_obs) == 0:
        p_obs = p_obs*np.ones(n_annotators)

    if n_groups == 1:
        collude_groups = False

    # Split annotators to into groups
    groups = np.array_split(np.arange(n_annotators), n_groups)
    
    if adversary_type == AdversaryType.RANDOM:
        responses = _gen_random_adversaries(
            gt_labels, groups, p_obs, p_attacked, collude_groups, same_resp_per_class, 
            same_resp_across_groups, rng
        )

    return responses[0] if n_groups == 1 else responses