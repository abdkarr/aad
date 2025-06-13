from pathlib import Path

import numpy as np

import aad 

PROJECT_DIR = str(Path(__file__).parents[1])
INPUT_DIR = Path(PROJECT_DIR, "_data", "inputs")

def test_majority_voting():
    response_mat, _ = aad.datasets.read_rte(INPUT_DIR)
    n_tasks = response_mat.shape[1]

    labels_hat = aad.majority_voting(response_mat)
    assert isinstance(labels_hat, np.ndarray)
    assert labels_hat.shape == (n_tasks, )
    assert not np.isin(0, labels_hat)

def test_dawid_skene():
    response_mat, _ = aad.datasets.read_rte(INPUT_DIR)
    n_tasks = response_mat.shape[1]

    ds_out = aad.dawid_skene(response_mat)

    labels_hat = ds_out["labels"]
    assert isinstance(labels_hat, np.ndarray)
    assert labels_hat.shape == (n_tasks, )
    assert not np.isin(0, labels_hat)