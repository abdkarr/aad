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

# Modify these locations as preferred
PROJECT_DIR = str(Path(__file__).parents[1])
INPUT_DIR = Path(PROJECT_DIR, "_data", "inputs")
OUTPUT_DIR = Path(PROJECT_DIR, "_data", "outputs")
FIGURE_DIR = Path(PROJECT_DIR, "_reports", "figures")

DATASET_LOADER = {"rte": aad.datasets.read_rte}

# Other simulation parameters
P_TARGET = {"rte": 0.0125}  # percentage of tasks targeted by adversaries
N_RUNS = 50


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--dataset",
    default="rte",
    type=click.Choice(list(DATASET_LOADER.keys())),
    show_default=True,
    help="Dataset to work on",
)
@click.option(
    "--p-adversaries",
    default=0.1,
    type=float,
    show_default=True,
    help="Fraction of adversaries to add the dataset",
)
def run(dataset: str, p_adversaries: float):

    # Read the original dataset, assumed to be clean
    honest_responses, gt_labels = DATASET_LOADER[dataset](INPUT_DIR)

    n_honests, n_tasks = honest_responses.shape
    # Average fraction of tasks observed per honest worker
    p_obs = np.mean(np.count_nonzero(honest_responses, axis=1) / n_tasks)

    # Setup simulation parameters
    n_adversaries = int(np.floor(p_adversaries * n_honests))
    p_target = P_TARGET[dataset]
    p_camo = p_obs
    n_workers = n_honests + n_adversaries

    worker_detect_perf = {"Run": [], "AUROC": [], "AUPRC": []}
    task_detect_perf = {"Run": [], "AUROC": [], "AUPRC": []}
    for r in range(N_RUNS):
        rng = np.random.default_rng()
        adv_responses, gt_targeted = aad.simulations.gen_adversaries(
            gt_labels, n_adversaries, p_target, p_camo, rng=rng
        )

        # Merge honest and adversarial responses
        response_mat = np.vstack([honest_responses, adv_responses])
        gt_adversaries = np.hstack([np.zeros(n_honests), np.ones(n_adversaries)])

        # Shuffle workers
        idx = np.arange(n_workers)
        rng.shuffle(idx)
        response_mat = response_mat[idx, :]
        gt_adversaries = gt_adversaries[idx]

        # Detection
        worker_scores, task_scores = aad.detect_attacks(response_mat, "weighted")

        worker_detect_perf["Run"].append(r)
        worker_detect_perf["AUROC"].append(auroc(gt_adversaries, worker_scores))
        worker_detect_perf["AUPRC"].append(auprc(gt_adversaries, worker_scores))
        task_detect_perf["Run"].append(r)
        task_detect_perf["AUROC"].append(auroc(gt_targeted, task_scores))
        task_detect_perf["AUPRC"].append(auprc(gt_targeted, task_scores))

    save_dir = Path(
        OUTPUT_DIR, dataset, f"p-adversaries-{p_adversaries:.2f}", "weighted"
    )
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
