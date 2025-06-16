import os

from pathlib import Path

import numpy as np

from scipy.io import savemat, loadmat

import aad

PROJECT_DIR = str(Path(__file__).parents[1])
INPUT_DIR = Path(PROJECT_DIR, "_data", "inputs")
OUTPUT_DIR = Path(PROJECT_DIR, "_data", "outputs")
FIGURE_DIR = Path(PROJECT_DIR, "_reports", "figures")

DATASET_LOADER = {"rte": aad.datasets.read_rte}


def gen_corrupted_data(dataset: str, exp_dir: os.PathLike | str, exp_params: dict):

    adv_file = Path(exp_dir, "adv_responses.mat")

    if adv_file.exists():
        data_dict = loadmat(adv_file, squeeze_me=True)
        adv_responses = data_dict["adv_responses"]
        gt_targeted = data_dict["gt_targeted"]
    else:
        honest_responses, gt_labels = DATASET_LOADER[dataset](INPUT_DIR)
        _, n_tasks = honest_responses.shape

        n_adversaries = exp_params["n_adversaries"]
        n_runs = exp_params["n_runs"]
        adv_responses = np.zeros((n_runs, n_adversaries, n_tasks))
        gt_targeted = np.zeros((n_runs, n_tasks))
        for r in range(exp_params["n_runs"]):
            rng = np.random.default_rng()
            r_responses, r_gt_targeted = aad.simulations.gen_adversaries(
                gt_labels,
                n_adversaries,
                exp_params["target_frac"],
                exp_params["camo_obs"],
                exp_params["target_obs"],
                exp_params["camo_reliability"],
                rng=rng,
            )

            adv_responses[r, :, :] = r_responses
            gt_targeted[r, :] = r_gt_targeted

        exp_dir.mkdir(parents=True, exist_ok=True)
        savemat(
            str(adv_file), {"adv_responses": adv_responses, "gt_targeted": gt_targeted}
        )

    return adv_responses, gt_targeted
