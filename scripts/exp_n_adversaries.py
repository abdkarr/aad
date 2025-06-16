"""
Experiment to observe the effect of number of adversaries on detection.
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auroc

import aad
import commons

# Other simulation parameters
TARGET_FRAC = {"rte": 0.0125}  # percentage of tasks targeted by adversaries
TARGET_OBS = 1
CAMO_RELIABILITY = 1
N_RUNS = 50

METHODS = ["weighted", "binary"]


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--dataset",
    default="rte",
    type=click.Choice(list(commons.DATASET_LOADER.keys())),
    show_default=True,
    help="Dataset to work on",
)
@click.option(
    "--adv-frac",
    default=0.1,
    type=float,
    show_default=True,
    help="Fraction of adversaries to add the dataset",
)
@click.option(
    "--method",
    default="binary",
    type=click.Choice(METHODS),
    show_default=True,
    help="Methods to use for detection",
)
def run(dataset: str, adv_frac: float, method: str):
    exp_dir = Path(commons.OUTPUT_DIR, dataset, "adv-frac", f"{adv_frac}")

    # Read the original dataset, assumed to be clean
    honest_responses, gt_labels = commons.DATASET_LOADER[dataset](commons.INPUT_DIR)
    n_honests, n_tasks = honest_responses.shape
    honest_obs = np.mean(np.count_nonzero(honest_responses, axis=1) / n_tasks)

    # Generate (or read if exists) simulation data
    exp_params = {
        "n_runs": N_RUNS,
        "n_adversaries": int(np.floor(adv_frac * n_honests)),
        "target_frac": TARGET_FRAC[dataset],
        "camo_obs": honest_obs,
        "target_obs": TARGET_OBS,
        "camo_reliability": CAMO_RELIABILITY,
    }
    adv_responses, gt_targeted = commons.gen_corrupted_data(
        dataset, exp_dir, exp_params
    )

    worker_detect_perf = {"Run": [], "AUROC": [], "AUPRC": []}
    task_detect_perf = {"Run": [], "AUROC": [], "AUPRC": []}
    for r in range(N_RUNS):
        n_adversaries = adv_responses[r, :, :].shape[0]
        n_workers = n_honests + n_adversaries
        rng = np.random.default_rng()

        # Merge honest and adversarial responses
        response_mat = np.vstack([honest_responses, adv_responses[r, :, :].squeeze()])
        gt_adversaries = np.hstack([np.zeros(n_honests), np.ones(n_adversaries)])

        # Shuffle workers
        idx = np.arange(n_workers)
        rng.shuffle(idx)
        response_mat = response_mat[idx, :]
        gt_adversaries = gt_adversaries[idx]

        # Detection
        worker_scores, task_scores = aad.detect_attacks(response_mat, method)

        worker_detect_perf["Run"].append(r)
        worker_detect_perf["AUROC"].append(auroc(gt_adversaries, worker_scores))
        worker_detect_perf["AUPRC"].append(auprc(gt_adversaries, worker_scores))
        
        task_detect_perf["Run"].append(r)
        task_detect_perf["AUROC"].append(auroc(gt_targeted[r, :], task_scores))
        task_detect_perf["AUPRC"].append(auprc(gt_targeted[r, :], task_scores))

    save_dir = Path(exp_dir, method)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = Path(save_dir, f"adversary-detection.csv")
    pd.DataFrame(worker_detect_perf).to_csv(save_file)

    save_file = Path(save_dir, f"targeted-task-detection.csv")
    pd.DataFrame(task_detect_perf).to_csv(save_file)


@cli.command()
def plot():
    pass


if __name__ == "__main__":
    cli()
