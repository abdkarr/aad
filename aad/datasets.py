import os
import tarfile

from pathlib import Path

import requests
import pandas as pd
import numpy as np
import numpy.typing as npt

from scipy import sparse


def read_rte(
    root_dir: os.PathLike | str, download: bool | None = True
) -> tuple[npt.NDArray, npt.NDArray]:
    """Read RTE dataset.

    This function looks for the data file `root_dir/rte/rte.standardized.tsv`
    to read RTE dataset. If this file does not exist and parameter `download`
    is True, it will attempt to download the file from remote server at 
    [here](https://web.archive.org/web/20230331023329/https://sites.google.com/site/nlpannotations/).

    Parameters
    ----------
    root_dir :
        The directory under which to look for `rte` folder
    download :
        Whether to download the data from remote server if
        `root_dir/rte/rte.standardized.tsv` is not found

    Returns
    -------
    response_mat :
        (M, N) dimensional matrix where `response_mat[i, j]` is the label provided
        by ith worker for jth task. `response_mat[i, j] = 0` if no label is provided
        by the ith worker for jth task.
    gt_labels :
        (N, ) dimensional vector where gt_labels[i] is the ground truth label of
        ith task.

    Raises
    ------
    Exception
        If remote server not available or `root_dir/rte/rte.standardized.tsv` is
        not found.
    """

    download_url = (
        "https://web.archive.org/web/20230331023329/"
        + "https://sites.google.com/site/nlpannotations/all_collected_data.tgz"
        + "?attredirects=0"
    )
    dataset_dir = Path(root_dir, "rte")
    dataset_file = Path(dataset_dir, "rte.standardized.tsv")

    # Check if RTE is already downloaded, if not download
    if (not dataset_file.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zipped_file = Path(dataset_dir, "all_collected_data.tgz")

        with requests.get(download_url, stream=True) as response:
            if response.status_code == requests.codes.ok:
                with open(zipped_file, mode="wb") as f:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        f.write(chunk)
            else:
                raise Exception("Remote not available to download RTE dataset.")

        # Unzip rte dataset from downloaded zip file, remove the rest
        with tarfile.open(zipped_file, "r:gz") as f:
            f.extractall(dataset_dir, members=["rte.standardized.tsv"], filter="data")

        zipped_file.unlink()

    try:
        raw_data = pd.read_csv(
            dataset_file,
            sep="\t",
            header=0,
            names=["annotation_id", "worker_id", "task_id", "response", "gt"],
        )
    except:
        raise Exception("`rte.standardized.tsv` file not found.")

    response_df = raw_data[["task_id", "worker_id", "response"]]
    gt_df = raw_data[["task_id", "gt"]].drop_duplicates(["task_id"])

    # Delete tasks without ground truth information from responses
    missing_tasks = np.setdiff1d(response_df["task_id"], gt_df["task_id"])
    missing_idx = np.isin(response_df["task_id"], missing_tasks)
    response_df = response_df.iloc[~missing_idx, :]

    # Create response matrix
    task_ids = response_df["task_id"].unique()
    worker_ids = response_df["worker_id"].unique()
    class_ids = response_df["response"].unique()

    n_tasks = len(task_ids)
    n_workers = len(worker_ids)

    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    worker_to_idx = {t: i for i, t in enumerate(worker_ids)}
    class_to_idx = {t: i + 1 for i, t in enumerate(class_ids)}

    rows = [worker_to_idx[t] for t in response_df["worker_id"]]
    cols = [task_to_idx[t] for t in response_df["task_id"]]
    vals = [class_to_idx[t] for t in response_df["response"]]

    response_mat = np.zeros((n_workers, n_tasks), dtype=np.int64)
    response_mat[rows, cols] = vals

    # Convert ground truth labels to class indices
    gt_labels = np.zeros(n_tasks, dtype=np.int64)
    for i, row in gt_df.iterrows():
        gt_labels[task_to_idx[row["task_id"]]] = class_to_idx[row["gt"]]

    return response_mat, gt_labels
