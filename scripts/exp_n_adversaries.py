"""
Experiment to observe the effect of number of adversaries on detection.
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

import commons

# Other simulation parameters
TARGET_FRAC = 0.2  # percentage of tasks targeted by adversaries
TARGET_OBS = 0.3
CAMO_RELIABILITY = 2
N_RUNS = 20

exp_name = f"adv-frac-tf{TARGET_FRAC:.2f}-to{TARGET_OBS:.2f}-cr{CAMO_RELIABILITY:.2f}"


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
    "--aggregator",
    default="wmv",
    show_default=True,
    type=click.Choice(["wmv", "wds"]),
    help="Methods to use for fusion step",
)
def run(dataset: str, adv_frac: float, aggregator: str):
    exp_dir = Path(commons.OUTPUT_DIR, dataset, exp_name, f"{adv_frac:.2f}")

    # Read the original dataset, assumed to be clean
    honest_responses, gt_labels = commons.DATASET_LOADER[dataset](commons.INPUT_DIR)
    n_honests, n_tasks = honest_responses.shape
    honest_obs = np.mean(np.count_nonzero(honest_responses, axis=1) / n_tasks)

    # Generate (or read if exists) simulation data
    exp_params = {
        "n_runs": N_RUNS,
        "n_adversaries": int(np.floor(adv_frac * n_honests)),
        "target_frac": TARGET_FRAC,
        "camo_obs": honest_obs,
        "target_obs": TARGET_OBS,
        "camo_reliability": CAMO_RELIABILITY,
    }
    adv_responses, _ = commons.gen_adversary_responses(
        dataset, exp_dir, exp_params
    )

    # Define output data structures
    fusion_perf = {"Run": [], "Method": [], "Accuracy": []}

    for r in range(N_RUNS):
        response_mat, _ = commons.gen_corrupted_data(
            honest_responses, adv_responses[r, :, :].squeeze()
        )
        labels_hat = commons.apply_proposed(response_mat, aggregator)

        # Performance calculation
        commons.update_aggregation_perf(
            gt_labels, labels_hat, fusion_perf, r, aggregator
        )

    # Save results
    save_dir = Path(exp_dir, aggregator)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = Path(save_dir, f"label-fusion.csv")
    pd.DataFrame(fusion_perf).to_csv(save_file)


if __name__ == "__main__":
    cli()
