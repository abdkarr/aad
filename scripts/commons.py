import os
import glob

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from scipy.io import savemat, loadmat
from sklearn import metrics

import aad
from aad import detect_attacks
from aad import dawid_skene, majority_voting
from aad.simulations import gen_adversaries
from aad.aggregators import weighted_dawid_skene, weighted_majority_voting

PROJECT_DIR = str(Path(__file__).parents[1])
INPUT_DIR = Path(PROJECT_DIR, "_data", "inputs")
OUTPUT_DIR = Path(PROJECT_DIR, "_data", "outputs")
FIGURE_DIR = Path(PROJECT_DIR, "_reports", "figures")

DATASET_LOADER = {
    "rte": aad.datasets.read_rte,
    "sp": lambda x: aad.datasets.read_mat_file(x, "sp"),
    "dog": lambda x: aad.datasets.read_mat_file(x, "dog"),
    "web": lambda x: aad.datasets.read_mat_file(x, "web"),
}

def gen_adversary_responses(dataset: str, exp_dir: os.PathLike | str, exp_params: dict):
    adv_file = Path(exp_dir, "adv_responses.mat")
    n_runs = exp_params.pop("n_runs")

    if adv_file.exists():
        data_dict = loadmat(adv_file, squeeze_me=True)
        adv_responses = data_dict["adv_responses"]
        gt_targeted = data_dict["gt_targeted"]
    else:
        # Read dataset
        honest_responses, gt_labels = DATASET_LOADER[dataset](INPUT_DIR)
        _, n_tasks = honest_responses.shape

        # Generate adversarial responses
        adv_responses = np.zeros((n_runs, exp_params["n_adversaries"], n_tasks))
        gt_targeted = np.zeros((n_runs, n_tasks))
        for r in range(n_runs):

            rng = np.random.default_rng()
            r_responses, r_gt_targeted = gen_adversaries(
                gt_labels, **exp_params, rng=rng
            )
            adv_responses[r, :, :] = r_responses
            gt_targeted[r, :] = r_gt_targeted

        # Save the outputs
        exp_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {"adv_responses": adv_responses, "gt_targeted": gt_targeted}
        savemat(str(adv_file), save_dict, do_compression=True)

    return adv_responses, gt_targeted


def gen_corrupted_data(honest_responses, adv_responses):
    n_honests = honest_responses.shape[0]
    n_adversaries = adv_responses.shape[0]
    n_workers = n_honests + n_adversaries
    rng = np.random.default_rng()

    # Merge honest and adversarial responses
    response_mat = np.vstack([honest_responses, adv_responses])
    gt_adversaries = np.hstack([np.zeros(n_honests), np.ones(n_adversaries)])

    # Shuffle workers
    idx = np.arange(n_workers)
    rng.shuffle(idx)
    response_mat = response_mat[idx, :]
    gt_adversaries = gt_adversaries[idx]

    return response_mat, gt_adversaries


def update_aggregation_perf(gt, pred, perf_dict, run, method):
    perf_dict["Run"].append(run)
    perf_dict["Method"].append(method)
    perf_dict["Accuracy"].append(metrics.accuracy_score(gt, pred))



def apply_proposed(response_mat, aggregator: str):
    worker_scores, task_scores = detect_attacks(response_mat, "weighted")

    if aggregator == "wmv":
        labels_hat = weighted_majority_voting._apply(
            response_mat, worker_scores, task_scores
        )
    elif aggregator == "wds":
        labels_hat = weighted_dawid_skene._apply(
            response_mat, worker_scores, task_scores, kind="weighted-EM"
        )["labels"]

    return labels_hat


def apply_greedypp(response_mat, method):
    n_workers, n_tasks = response_mat.shape
    biadj_mat = aad.detection._construct_biadj_mat(response_mat, "weighted")
    adj = np.block(
        [
            [np.zeros((n_workers, n_workers)), biadj_mat],
            [biadj_mat.T, np.zeros((n_tasks, n_tasks))],
        ]
    )
    G = nx.from_numpy_array(adj)
    densest_sg = np.array(
        list(nx.approximation.densest_subgraph(G, method="greedy++", iterations=10)[1])
    )

    worker_scores = np.zeros(n_workers)
    worker_scores[densest_sg[densest_sg < n_workers]] = 1

    task_scores = np.zeros(n_tasks)
    task_scores[densest_sg[densest_sg >= n_workers] - n_workers] = 1

    aggregator = method.split("-")[1]
    if aggregator == "wmv":
        labels_hat = weighted_majority_voting._apply(
            response_mat, worker_scores, task_scores
        )
    elif aggregator == "dswmv":
        labels_hat = weighted_dawid_skene._apply(
            response_mat, worker_scores, task_scores, kind="original-EM"
        )["labels"]
    elif aggregator == "wdswmv":
        labels_hat = weighted_dawid_skene._apply(
            response_mat, worker_scores, task_scores, kind="weighted-EM"
        )["labels"]

    return worker_scores, task_scores, labels_hat


def apply_methods(response_mat, method: str):

    if method.startswith("weighted") or method.startswith("binary"):
        worker_scores, task_scores, labels_hat = apply_proposed(response_mat, method)
    elif method == "ds":
        labels_hat = aad.dawid_skene(response_mat)["labels"]
        worker_scores = None
        task_scores = None
    elif method == "mv":
        labels_hat = aad.majority_voting(response_mat)
        worker_scores = None
        task_scores = None

    return {
        "worker_scores": worker_scores,
        "task_scores": task_scores,
        "labels_hat": labels_hat,
    }


def read_detection_results(exp_name):
    adversary_detection = {}
    target_detection = {}
    for dataset in DATASET_LOADER.keys():
        exp_dir = Path(OUTPUT_DIR, dataset, exp_name)

        adversary_detection[dataset] = []
        target_detection[dataset] = []
        for exp_case in glob.glob(str(Path(exp_dir, "*"))):
            exp_path = Path(exp_case)
            param_val = float(exp_path.name)

            # read proposed
            for method in ["binary-wmv", "weighted-wmv"]:
                fname = Path(exp_path, method, "adversary-detection.csv")
                adversary_detection[dataset].append(pd.read_csv(fname, index_col=0))
                adversary_detection[dataset][-1]["Param"] = param_val
                adversary_detection[dataset][-1]["Style"] = 0

                fname = Path(exp_path, method, "target-detection.csv")
                target_detection[dataset].append(pd.read_csv(fname, index_col=0))
                target_detection[dataset][-1]["Param"] = param_val
                target_detection[dataset][-1]["Style"] = 0

            # read dacs
            fname = Path(exp_path, "dacs", "adversary-detection.csv")
            adversary_detection[dataset].append(pd.read_csv(fname, index_col=0))
            adversary_detection[dataset][-1]["Param"] = param_val
            adversary_detection[dataset][-1]["Style"] = 1

            # read mmsr
            fname = Path(exp_path, "mmsr", "adversary-detection.mat")
            data_dict = loadmat(fname, squeeze_me=True)
            n_runs = len(data_dict["auroc"])
            auroc_df = {"Run": np.arange(n_runs), "AUROC": data_dict["auroc"]}
            adversary_detection[dataset].append(pd.DataFrame(auroc_df))
            adversary_detection[dataset][-1]["Method"] = "mmsr"
            adversary_detection[dataset][-1]["Param"] = param_val
            adversary_detection[dataset][-1]["Style"] = 1

        adversary_detection[dataset] = pd.concat(adversary_detection[dataset])
        target_detection[dataset] = pd.concat(target_detection[dataset])

    return adversary_detection, target_detection


def read_label_fusion(exp_name, methods):
    label_fusion = {}
    for dataset in DATASET_LOADER.keys():
        exp_dir = Path(OUTPUT_DIR, dataset, exp_name)

        label_fusion[dataset] = []
        for exp_case in glob.glob(str(Path(exp_dir, "*"))):
            exp_path = Path(exp_case)
            param_val = float(exp_path.name)

            for method in methods:
                fname = Path(exp_path, method, "label-fusion.csv")
                label_fusion[dataset].append(pd.read_csv(fname, index_col=0))
                label_fusion[dataset][-1]["Param"] = param_val

                # Line style based on type of the method
                if method.startswith("weighted"):
                    label_fusion[dataset][-1]["Style"] = 0  # Proposed
                if method in ["ds", "mv"]:
                    label_fusion[dataset][-1]["Style"] = 2  # Others
                if method in ["mmsr-ds", "mmsr-mv", "dacs"]:
                    label_fusion[dataset][-1]["Style"] = 1  # Anomaly aware

        label_fusion[dataset] = pd.concat(label_fusion[dataset])

    return label_fusion
